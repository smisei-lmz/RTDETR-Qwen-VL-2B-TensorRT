#pragma once
#include "v4l2_camera.hpp"
#include <nvbufsurface.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <cuda.h>
#include <cudaEGL.h>
#include <cuda_runtime.h>
namespace edge {
inline void nvcheck(int status,const char*name){if(status!=0)throw std::runtime_error(std::string(name)+" failed: "+std::to_string(status));}
inline void rtcheck(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
struct NvImage {
 NvBufSurface*surface=nullptr;CUgraphicsResource resource=nullptr;CUeglFrame frame{};
 cudaTextureObject_t ty=0,tuv=0;bool mapped=false;
 ~NvImage(){if(ty)cudaDestroyTextureObject(ty);if(tuv)cudaDestroyTextureObject(tuv);if(resource)cuGraphicsUnregisterResource(resource);
  if(mapped)NvBufSurfaceUnMapEglImage(surface,0);if(surface)NvBufSurfaceDestroy(surface);}
};
class NvPool {
 EGLDisplay display=EGL_NO_DISPLAY;std::vector<std::unique_ptr<NvImage>> images;v4l2_pix_format fmt;
 static cudaTextureObject_t texture(CUarray input,bool chroma){
  auto array=reinterpret_cast<cudaArray_t>(input);cudaChannelFormatDesc channel{};rtcheck(cudaGetChannelDesc(&channel,array));
  if(channel.f!=cudaChannelFormatKindUnsigned||channel.x!=8||channel.y!=(chroma?8:0)||channel.z||channel.w)
    throw std::runtime_error("unexpected CUDA array channel layout; cannot interpret as NV12");
  cudaResourceDesc resource{};resource.resType=cudaResourceTypeArray;resource.res.array.array=array;
  cudaTextureDesc desc{};desc.addressMode[0]=cudaAddressModeClamp;desc.addressMode[1]=cudaAddressModeClamp;
  desc.filterMode=cudaFilterModePoint;desc.readMode=cudaReadModeElementType;desc.normalizedCoords=0;
  cudaTextureObject_t result=0;rtcheck(cudaCreateTextureObject(&result,&resource,&desc,nullptr));return result;
 }
public:
 NvPool(int count,v4l2_pix_format f):fmt(f){
  rtcheck(cudaSetDevice(0));rtcheck(cudaFree(nullptr)); // Runtime primary context, shared with the backend.
  display=eglGetDisplay(EGL_DEFAULT_DISPLAY);if(display==EGL_NO_DISPLAY||!eglInitialize(display,nullptr,nullptr))throw std::runtime_error("EGL initialization failed; inspect Jetson headless/display configuration");
  try{for(int i=0;i<count;i++){
   auto item=std::make_unique<NvImage>();NvBufSurfaceCreateParams p{};p.gpuId=0;p.width=f.width;p.height=f.height;
   p.colorFormat=f.pixelformat==V4L2_PIX_FMT_NV12?NVBUF_COLOR_FORMAT_NV12:NVBUF_COLOR_FORMAT_YUYV;
   p.layout=NVBUF_LAYOUT_PITCH;p.memType=NVBUF_MEM_SURFACE_ARRAY;
   nvcheck(NvBufSurfaceCreate(&item->surface,1,&p),"NvBufSurfaceCreate");auto&s=item->surface->surfaceList[0];
   if(s.pitch!=f.bytesperline||s.dataSize<f.sizeimage)throw std::runtime_error("camera stride/size is incompatible with allocated NVIDIA surface");
   if(f.pixelformat==V4L2_PIX_FMT_NV12&&(s.planeParams.num_planes!=2||s.planeParams.offset[1]!=f.bytesperline*f.height||s.planeParams.pitch[1]!=f.bytesperline))
    throw std::runtime_error("single-plane V4L2 NV12 offsets do not match NVIDIA surface; cannot safely import");
   nvcheck(NvBufSurfaceMapEglImage(item->surface,0),"NvBufSurfaceMapEglImage");item->mapped=true;
   nvcheck(cuGraphicsEGLRegisterImage(&item->resource,static_cast<EGLImageKHR>(s.mappedAddr.eglImage),CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY),"cuGraphicsEGLRegisterImage");
   nvcheck(cuGraphicsResourceGetMappedEglFrame(&item->frame,item->resource,0,0),"cuGraphicsResourceGetMappedEglFrame");
   if(item->frame.frameType==CU_EGL_FRAME_TYPE_ARRAY){
    if(f.pixelformat!=V4L2_PIX_FMT_NV12||item->frame.planeCount<2)throw std::runtime_error("array import supports validated two-plane NV12 only; YUYV needs pitch mapping");
    item->ty=texture(item->frame.frame.pArray[0],false);item->tuv=texture(item->frame.frame.pArray[1],true);
   }else if(item->frame.frameType!=CU_EGL_FRAME_TYPE_PITCH)throw std::runtime_error("unsupported EGL frame storage");
   images.push_back(std::move(item));
  }}catch(...){images.clear();eglTerminate(display);display=EGL_NO_DISPLAY;throw;}
 }
 ~NvPool(){images.clear();if(display!=EGL_NO_DISPLAY)eglTerminate(display);}
 int fd(int index)const{return int(images.at(index)->surface->surfaceList[0].bufferDesc);}
 size_t bytes(int index)const{return images.at(index)->surface->surfaceList[0].dataSize;}
 DeviceYuv view(int index,int matrix,bool full)const{
  auto&i=*images.at(index);auto&s=i.surface->surfaceList[0];DeviceYuv v{};
  v.w=fmt.width;v.h=fmt.height;v.pitch_y=s.planeParams.pitch[0];v.pitch_uv=fmt.pixelformat==V4L2_PIX_FMT_NV12?s.planeParams.pitch[1]:0;
  v.format=fmt.pixelformat==V4L2_PIX_FMT_NV12?0:1;v.matrix=matrix;v.full_range=full;v.tex_y=i.ty;v.tex_uv=i.tuv;
  if(!v.tex_y){v.y=static_cast<const uint8_t*>(i.frame.frame.pPitch[0]);v.uv=v.format==0?static_cast<const uint8_t*>(i.frame.frame.pPitch[1]):nullptr;
   if(!v.y||(v.format==0&&!v.uv))throw std::runtime_error("missing EGL pitch plane");}
  return v;
 }
 Image cpu_reference(int index,int matrix,bool full){auto&i=*images.at(index);auto&s=i.surface->surfaceList[0];
  nvcheck(NvBufSurfaceMap(i.surface,0,-1,NVBUF_MAP_READ),"NvBufSurfaceMap CPU");
  try{nvcheck(NvBufSurfaceSyncForCpu(i.surface,0,-1),"NvBufSurfaceSyncForCpu");
   std::vector<uint8_t> linear(fmt.sizeimage);size_t ybytes=size_t(fmt.bytesperline)*fmt.height;
   std::memcpy(linear.data(),s.mappedAddr.addr[0],ybytes);
   if(fmt.pixelformat==V4L2_PIX_FMT_NV12)std::memcpy(linear.data()+ybytes,s.mappedAddr.addr[1],ybytes/2);
   Image image=cpu_yuv(linear.data(),fmt,matrix,full);nvcheck(NvBufSurfaceUnMap(i.surface,0,-1),"NvBufSurfaceUnMap CPU");return image;
  }catch(...){NvBufSurfaceUnMap(i.surface,0,-1);throw;}}
};
}
