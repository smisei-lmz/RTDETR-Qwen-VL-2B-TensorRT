#pragma once
#ifndef __linux__
#error This backend requires Linux V4L2.
#endif
#include "edge.hpp"
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <poll.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>

namespace edge {
inline int xioctl(int fd,unsigned long request,void*arg){int r;do{r=ioctl(fd,request,arg);}while(r<0&&errno==EINTR);return r;}
inline void oscheck(int value,const char*operation){if(value<0)throw std::runtime_error(std::string(operation)+": "+std::strerror(errno));}
struct Captured {uint32_t index,sequence,flags,bytes;uint64_t generation,driver_ns;};
class Camera {
 struct Buffer {void*data=MAP_FAILED;size_t bytes=0;int fd=-1;bool queued=false;uint64_t generation=0;};
 int fd=-1;bool streaming=false;v4l2_memory memory=V4L2_MEMORY_MMAP;
 std::vector<Buffer> buffers;
public:
 v4l2_pix_format format{};
 Camera(const std::string&device,int w,int h,uint32_t fourcc){
  fd=open(device.c_str(),O_RDWR|O_NONBLOCK|O_CLOEXEC);oscheck(fd,"open camera");
  try{v4l2_capability c{};oscheck(xioctl(fd,VIDIOC_QUERYCAP,&c),"QUERYCAP");
   uint32_t cap=(c.capabilities&V4L2_CAP_DEVICE_CAPS)?c.device_caps:c.capabilities;
   if(!(cap&V4L2_CAP_VIDEO_CAPTURE)||!(cap&V4L2_CAP_STREAMING))throw std::runtime_error("This demo requires single-planar CAPTURE + STREAMING; MPLANE needs an adapter");
   v4l2_format f{};f.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;f.fmt.pix.width=w;f.fmt.pix.height=h;f.fmt.pix.pixelformat=fourcc;f.fmt.pix.field=V4L2_FIELD_NONE;
   oscheck(xioctl(fd,VIDIOC_S_FMT,&f),"S_FMT");format=f.fmt.pix;
   if(format.pixelformat!=fourcc||format.width%2||format.height%2||format.width==0||format.height==0||format.width>4096||format.height>4096)throw std::runtime_error("Unsupported negotiated format/dimensions");
   if(fourcc!=V4L2_PIX_FMT_YUYV&&fourcc!=V4L2_PIX_FMT_NV12)throw std::runtime_error("Only uncompressed YUYV/NV12 supported");
   if(format.bytesperline<format.width*(fourcc==V4L2_PIX_FMT_YUYV?2:1))throw std::runtime_error("invalid stride");
  }catch(...){close(fd);fd=-1;throw;}
 }
 ~Camera(){stop();for(auto&b:buffers){if(b.data!=MAP_FAILED)munmap(b.data,b.bytes);if(b.fd>=0)close(b.fd);}if(fd>=0)close(fd);}
 int allocate(int count,bool import=false,bool export_fd=false){
  memory=import?V4L2_MEMORY_DMABUF:V4L2_MEMORY_MMAP;v4l2_requestbuffers req{};
  req.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;req.memory=memory;req.count=count;
  oscheck(xioctl(fd,VIDIOC_REQBUFS,&req),"REQBUFS");if(req.count<2)throw std::runtime_error("fewer than two capture buffers");buffers.resize(req.count);
  if(!import)for(uint32_t i=0;i<req.count;i++){
   v4l2_buffer b{};b.type=req.type;b.memory=memory;b.index=i;oscheck(xioctl(fd,VIDIOC_QUERYBUF,&b),"QUERYBUF");
   buffers[i].bytes=b.length;buffers[i].data=mmap(nullptr,b.length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,b.m.offset);
   if(buffers[i].data==MAP_FAILED)throw std::runtime_error("mmap failed");
   if(export_fd){v4l2_exportbuffer e{};e.type=req.type;e.index=i;e.flags=O_CLOEXEC|O_RDWR;oscheck(xioctl(fd,VIDIOC_EXPBUF,&e),"EXPBUF");buffers[i].fd=e.fd;}
  }return req.count;
 }
 void queue(uint32_t index,int imported_fd=-1,size_t length=0){
  auto&slot=buffers.at(index);if(slot.queued)throw std::runtime_error("double QBUF");v4l2_buffer b{};
  b.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;b.memory=memory;b.index=index;
  if(memory==V4L2_MEMORY_DMABUF){if(imported_fd<0)throw std::runtime_error("missing imported fd");b.m.fd=imported_fd;b.length=length;}
  oscheck(xioctl(fd,VIDIOC_QBUF,&b),"QBUF");slot.queued=true;
 }
 void release(const Captured&c,int imported_fd=-1,size_t length=0){if(buffers.at(c.index).generation!=c.generation)throw std::runtime_error("stale camera return token");queue(c.index,imported_fd,length);}
 void start(){v4l2_buf_type type=V4L2_BUF_TYPE_VIDEO_CAPTURE;oscheck(xioctl(fd,VIDIOC_STREAMON,&type),"STREAMON");streaming=true;}
 void stop(){if(streaming){v4l2_buf_type type=V4L2_BUF_TYPE_VIDEO_CAPTURE;xioctl(fd,VIDIOC_STREAMOFF,&type);streaming=false;}}
 std::optional<Captured> next(int timeout_ms=5){pollfd p{fd,POLLIN,0};int result=poll(&p,1,timeout_ms);
  if(result<0&&errno==EINTR)return {};oscheck(result,"poll");if(result==0)return {};
  if(p.revents&(POLLERR|POLLHUP|POLLNVAL))throw std::runtime_error("camera poll error/disconnect");
  v4l2_buffer b{};b.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;b.memory=memory;
  if(xioctl(fd,VIDIOC_DQBUF,&b)<0){if(errno==EAGAIN)return {};oscheck(-1,"DQBUF");}
  auto&s=buffers.at(b.index);if(!s.queued)throw std::runtime_error("DQBUF ownership mismatch");s.queued=false;s.generation++;
  return Captured{b.index,b.sequence,b.flags,b.bytesused,s.generation,uint64_t(b.timestamp.tv_sec)*1000000000ULL+uint64_t(b.timestamp.tv_usec)*1000};
 }
 bool valid(const Captured&c)const{
  uint64_t min_bytes=uint64_t(format.bytesperline)*format.height;
  if(format.pixelformat==V4L2_PIX_FMT_NV12)min_bytes+=uint64_t(format.bytesperline)*(format.height/2);
  return !(c.flags&V4L2_BUF_FLAG_ERROR)&&c.bytes>=min_bytes&&c.bytes<=format.sizeimage
      &&(memory==V4L2_MEMORY_DMABUF||c.bytes<=buffers.at(c.index).bytes);
 }
 const uint8_t*data(uint32_t index)const{return static_cast<const uint8_t*>(buffers.at(index).data);}
 int exported_fd(uint32_t index)const{return buffers.at(index).fd;}
};
inline Image cpu_yuv(const uint8_t*data,const v4l2_pix_format&f,int matrix,bool full){
 Image im{int(f.width),int(f.height),std::vector<uint8_t>(size_t(f.width)*f.height*3)};
 for(unsigned y=0;y<f.height;y++)for(unsigned x=0;x<f.width;x++){
  float yy,u,v;if(f.pixelformat==V4L2_PIX_FMT_NV12){yy=data[y*f.bytesperline+x];auto p=data+f.bytesperline*f.height+(y/2)*f.bytesperline+(x/2)*2;u=p[0]-128.f;v=p[1]-128.f;}
  else{auto p=data+y*f.bytesperline+(x/2)*4;yy=p[(x%2)*2];u=p[1]-128.f;v=p[3]-128.f;}
  float r,g,b;if(full){if(matrix==709){r=yy+1.5748f*v;g=yy-.187324f*u-.468124f*v;b=yy+1.8556f*u;}else{r=yy+1.402f*v;g=yy-.344136f*u-.714136f*v;b=yy+1.772f*u;}}
  else{yy=1.164383f*(yy-16);if(matrix==709){r=yy+1.792741f*v;g=yy-.213249f*u-.532909f*v;b=yy+2.112402f*u;}else{r=yy+1.596027f*v;g=yy-.391762f*u-.812968f*v;b=yy+2.017232f*u;}}
  auto clip=[](float z){return uint8_t(std::min(255.f,std::max(0.f,z))+.5f);};size_t i=(size_t(y)*f.width+x)*3;im.rgb[i]=clip(r);im.rgb[i+1]=clip(g);im.rgb[i+2]=clip(b);
 }return im;
}
}
