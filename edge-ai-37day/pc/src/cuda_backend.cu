#include "edge.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <iostream>
#ifdef EDGE_TENSORRT
#include <NvInfer.h>
#endif
namespace edge {
static void ck(cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
template<class T>__device__ T cast_value(float x){return T(x);}
template<>__device__ __half cast_value<__half>(float x){return __float2half(x);}
template<class T>__device__ float read_value(T x){return float(x);}
template<>__device__ float read_value<__half>(__half x){return __half2float(x);}
template<class T>__global__ void prep(const uint8_t*src,int sw,int sh,T*dst,int dw,int dh){
 int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=dw*dh)return;int x=index%dw,y=index/dw;
 for(int c=0;c<3;c++){
   float fx=fmaxf(0,fminf(sw-1,(x+.5f)*sw/dw-.5f)),fy=fmaxf(0,fminf(sh-1,(y+.5f)*sh/dh-.5f));
   int x0=int(fx),y0=int(fy),x1=min(x0+1,sw-1),y1=min(y0+1,sh-1);float ax=fx-x0,ay=fy-y0;
   float a=src[(y0*sw+x0)*3+c],b=src[(y0*sw+x1)*3+c],d=src[(y1*sw+x0)*3+c],e=src[(y1*sw+x1)*3+c];
   float value=(1-ay)*((1-ax)*a+ax*b)+ay*((1-ax)*d+ax*e);
   dst[c*dw*dh+index]=cast_value<T>(value/255);
 }
}
__global__ void color_bounds(const uint8_t*rgb,int w,int h,int*out){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=w*h)return;
 if(rgb[i*3]>160&&rgb[i*3+1]<100&&rgb[i*3+2]<100){int x=i%w,y=i/w;atomicMin(out,x);atomicMin(out+1,y);atomicMax(out+2,x);atomicMax(out+3,y);}}
__global__ void yuv_rgb(DeviceYuv src,uint8_t*dst){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=src.w*src.h)return;int x=i%src.w,y=i/src.w;float yy,u,v;
 if(src.format==0){if(src.tex_y){yy=tex2D<unsigned char>(src.tex_y,x+.5f,y+.5f);auto uv=tex2D<uchar2>(src.tex_uv,x/2+.5f,y/2+.5f);u=uv.x-128.f;v=uv.y-128.f;}
 else{yy=src.y[y*src.pitch_y+x];auto p=src.uv+(y/2)*src.pitch_uv+(x/2)*2;u=p[0]-128.f;v=p[1]-128.f;}}
 else{auto p=src.y+y*src.pitch_y+(x/2)*4;yy=p[(x%2)*2];u=p[1]-128.f;v=p[3]-128.f;}
 float r,g,b;if(src.full_range){if(src.matrix==709){r=yy+1.5748f*v;g=yy-.187324f*u-.468124f*v;b=yy+1.8556f*u;}
 else{r=yy+1.402f*v;g=yy-.344136f*u-.714136f*v;b=yy+1.772f*u;}}
 else{yy=1.164383f*(yy-16);if(src.matrix==709){r=yy+1.792741f*v;g=yy-.213249f*u-.532909f*v;b=yy+2.112402f*u;}
 else{r=yy+1.596027f*v;g=yy-.391762f*u-.812968f*v;b=yy+2.017232f*u;}}
 dst[i*3]=uint8_t(fminf(255,fmaxf(0,r))+.5f);dst[i*3+1]=uint8_t(fminf(255,fmaxf(0,g))+.5f);dst[i*3+2]=uint8_t(fminf(255,fmaxf(0,b))+.5f);
}
#ifdef EDGE_TENSORRT
class Logger:public nvinfer1::ILogger{void log(Severity s,const char*m)noexcept override{if(s<=Severity::kWARNING)std::cerr<<"TRT: "<<m<<'\n';}};
#endif
struct Slot {
 cudaStream_t stream=nullptr;cudaEvent_t start=nullptr,source_done=nullptr,done=nullptr,copy_end=nullptr,prep_end=nullptr,infer_start=nullptr,infer_end=nullptr;
 uint8_t *host=nullptr,*raw=nullptr;void *input=nullptr,*output=nullptr,*check=nullptr,*host_output=nullptr;
 int *bounds=nullptr,*host_bounds=nullptr;
 bool busy=false,recorded=false,external=false;Meta meta;std::vector<float> reference;
#ifdef EDGE_TENSORRT
 std::unique_ptr<nvinfer1::IExecutionContext> context;
#endif
 ~Slot(){if(stream)cudaStreamSynchronize(stream);
#ifdef EDGE_TENSORRT
 context.reset();
#endif
 if(input)cudaFree(input);if(output)cudaFree(output);if(raw)cudaFree(raw);if(bounds)cudaFree(bounds);
 if(host_output)cudaFreeHost(host_output);if(host)cudaFreeHost(host);if(host_bounds)cudaFreeHost(host_bounds);if(check)cudaFreeHost(check);
 if(start)cudaEventDestroy(start);if(source_done)cudaEventDestroy(source_done);if(done)cudaEventDestroy(done);
 if(copy_end)cudaEventDestroy(copy_end);if(prep_end)cudaEventDestroy(prep_end);if(infer_start)cudaEventDestroy(infer_start);if(infer_end)cudaEventDestroy(infer_end);if(stream)cudaStreamDestroy(stream);}
};
class Cuda:public Backend {
 Settings cfg;int iw=640,ih=640,anchors=0,attrs=0;bool input_half=false,output_half=false;
 size_t input_bytes=0,output_bytes=0;
#ifdef EDGE_TENSORRT
 Logger logger;std::unique_ptr<nvinfer1::IRuntime> runtime;std::unique_ptr<nvinfer1::ICudaEngine> engine;
 std::string input_name,output_name;
#endif
 std::vector<std::unique_ptr<Slot>> slots;
 void launch(Slot&s){
  int n=iw*ih; // DETR stretches; no padding
  if(input_half)prep<<<(n+255)/256,256,0,s.stream>>>(s.raw,s.meta.w,s.meta.h,static_cast<__half*>(s.input),iw,ih);
  else prep<<<(n+255)/256,256,0,s.stream>>>(s.raw,s.meta.w,s.meta.h,static_cast<float*>(s.input),iw,ih);
  ck(cudaGetLastError());ck(cudaEventRecord(s.prep_end,s.stream));
  if(cfg.verify)ck(cudaMemcpyAsync(s.check,s.input,input_bytes,cudaMemcpyDeviceToHost,s.stream));
  if(s.external&&s.meta.snapshot_requested)ck(cudaMemcpyAsync(s.host,s.raw,size_t(s.meta.w)*s.meta.h*3,cudaMemcpyDeviceToHost,s.stream));
  ck(cudaEventRecord(s.infer_start,s.stream));
  if(cfg.backend=="trt"){
#ifdef EDGE_TENSORRT
   if(!s.context->enqueueV3(s.stream))throw std::runtime_error("enqueueV3 failed after preprocessing submission");
   ck(cudaEventRecord(s.infer_end,s.stream));
   ck(cudaMemcpyAsync(s.host_output,s.output,output_bytes,cudaMemcpyDeviceToHost,s.stream));
#endif
  }else{
   s.host_bounds[0]=s.meta.w;s.host_bounds[1]=s.meta.h;s.host_bounds[2]=-1;s.host_bounds[3]=-1;
   ck(cudaMemcpyAsync(s.bounds,s.host_bounds,4*sizeof(int),cudaMemcpyHostToDevice,s.stream));
   int pixels=s.meta.w*s.meta.h;color_bounds<<<(pixels+255)/256,256,0,s.stream>>>(s.raw,s.meta.w,s.meta.h,s.bounds);
   ck(cudaGetLastError());ck(cudaEventRecord(s.infer_end,s.stream));ck(cudaMemcpyAsync(s.host_bounds,s.bounds,4*sizeof(int),cudaMemcpyDeviceToHost,s.stream));
  }
  ck(cudaEventRecord(s.done,s.stream));s.recorded=true;
 }
 void begin(Slot&s,Meta meta,bool external){if(s.busy)throw std::runtime_error("GPU lane still busy");
  if(meta.w<1||meta.h<1||int64_t(meta.w)*meta.h>cfg.max_pixels)throw std::runtime_error("source larger than --max-pixels");
  if(external&&cfg.verify)throw std::runtime_error("external YUV parity needs a separately synchronized CPU reference; disable --verify-preprocess");
  s.meta=std::move(meta);s.busy=true;s.recorded=false;s.external=external;}
public:
 explicit Cuda(Settings s):cfg(s){
  ck(cudaSetDevice(0));
  if(cfg.backend!="cuda"&&cfg.backend!="trt")throw std::runtime_error("unknown GPU backend");
  if(cfg.backend=="trt"){
#ifdef EDGE_TENSORRT
   std::ifstream f(cfg.engine,std::ios::binary);if(!f)throw std::runtime_error("missing --engine file");
   std::vector<char>bytes((std::istreambuf_iterator<char>(f)),{});runtime.reset(nvinfer1::createInferRuntime(logger));
   if(!runtime)throw std::runtime_error("createInferRuntime failed");engine.reset(runtime->deserializeCudaEngine(bytes.data(),bytes.size()));
   if(!engine)throw std::runtime_error("engine incompatible/corrupt");
   for(int i=0;i<engine->getNbIOTensors();i++){const char*name=engine->getIOTensorName(i);
    auto mode=engine->getTensorIOMode(name);auto&dest=mode==nvinfer1::TensorIOMode::kINPUT?input_name:output_name;
    if(!dest.empty())throw std::runtime_error("only one input and one RT-DETR deployment output supported");dest=name;}
   if(input_name.empty()||output_name.empty())throw std::runtime_error("missing model I/O");
   auto id=engine->getTensorShape(input_name.c_str()),od=engine->getTensorShape(output_name.c_str());
   if(input_name!="images"||output_name!="rtdetr_detections")
    throw std::runtime_error("wrong engine contract: rebuild with tools/model_tools.py engine for RT-DETRv2");
   if(id.nbDims!=4||id.d[0]!=1||id.d[1]!=3||id.d[2]!=640||id.d[3]!=640||
      od.nbDims!=3||od.d[0]!=1||od.d[1]!=300||od.d[2]!=6)
    throw std::runtime_error("require images [1,3,640,640] and RT-DETR [1,300,6] normalized xyxy/score/class");
   iw=id.d[3];ih=id.d[2];anchors=od.d[1];attrs=6;
   if(engine->getTensorFormat(input_name.c_str())!=nvinfer1::TensorFormat::kLINEAR||engine->getTensorFormat(output_name.c_str())!=nvinfer1::TensorFormat::kLINEAR)
    throw std::runtime_error("only LINEAR tensor I/O supported");
   auto it=engine->getTensorDataType(input_name.c_str()),ot=engine->getTensorDataType(output_name.c_str());
   for(auto type:{it,ot})if(type!=nvinfer1::DataType::kFLOAT&&type!=nvinfer1::DataType::kHALF)throw std::runtime_error("only FP32/FP16 I/O supported");
   input_half=it==nvinfer1::DataType::kHALF;output_half=ot==nvinfer1::DataType::kHALF;
   output_bytes=size_t(attrs)*anchors*(output_half?2:4);
#else
   throw std::runtime_error("build with -DEDGE_WITH_TENSORRT=ON");
#endif
  }
  input_bytes=size_t(iw)*ih*3*(input_half?2:4);
  for(int i=0;i<cfg.lanes;i++){
   auto p=std::make_unique<Slot>();ck(cudaStreamCreateWithFlags(&p->stream,cudaStreamNonBlocking));
   ck(cudaEventCreate(&p->start));ck(cudaEventCreateWithFlags(&p->source_done,cudaEventDisableTiming));ck(cudaEventCreate(&p->done));
   ck(cudaEventCreate(&p->copy_end));ck(cudaEventCreate(&p->prep_end));ck(cudaEventCreate(&p->infer_start));ck(cudaEventCreate(&p->infer_end));
   ck(cudaMallocHost(reinterpret_cast<void**>(&p->host),size_t(cfg.max_pixels)*3));ck(cudaMalloc(reinterpret_cast<void**>(&p->raw),size_t(cfg.max_pixels)*3));
   ck(cudaMalloc(&p->input,input_bytes));if(cfg.verify)ck(cudaMallocHost(&p->check,input_bytes));
   ck(cudaMalloc(reinterpret_cast<void**>(&p->bounds),4*sizeof(int)));ck(cudaMallocHost(reinterpret_cast<void**>(&p->host_bounds),4*sizeof(int)));
   if(cfg.backend=="trt"){
#ifdef EDGE_TENSORRT
    ck(cudaMalloc(&p->output,output_bytes));
    ck(cudaMallocHost(&p->host_output,output_bytes));
    p->context.reset(engine->createExecutionContext());if(!p->context)throw std::runtime_error("execution context allocation failed");
    if(!p->context->setTensorAddress(input_name.c_str(),p->input)||!p->context->setTensorAddress(output_name.c_str(),p->output))throw std::runtime_error("binding failed");
#endif
   }slots.push_back(std::move(p));
  }
 }
 int available()override{for(size_t i=0;i<slots.size();i++)if(!slots[i]->busy)return int(i);return -1;}
 void submit(int lane,View src,Meta meta)override{
  auto&s=*slots.at(lane);begin(s,std::move(meta),false);std::memcpy(s.host,src.rgb,size_t(src.w)*src.h*3);
  if(cfg.verify)s.reference=preprocess_reference(src,iw,ih);
  ck(cudaEventRecord(s.start,s.stream));
  ck(cudaMemcpyAsync(s.raw,s.host,size_t(src.w)*src.h*3,cudaMemcpyHostToDevice,s.stream));
  ck(cudaEventRecord(s.copy_end,s.stream));
  ck(cudaEventRecord(s.source_done,s.stream));launch(s);
 }
 void submit_device(int lane,DeviceYuv src,Meta meta)override{
  auto&s=*slots.at(lane);begin(s,std::move(meta),true);int n=src.w*src.h;ck(cudaEventRecord(s.start,s.stream));
  yuv_rgb<<<(n+255)/256,256,0,s.stream>>>(src,s.raw);ck(cudaGetLastError());
  ck(cudaEventRecord(s.copy_end,s.stream));
  // Camera can be returned after this event: all further work uses our own RGB buffer.
  ck(cudaEventRecord(s.source_done,s.stream));launch(s);
 }
 bool source_ready(int lane)override{auto&s=*slots.at(lane);if(!s.recorded)return false;auto status=cudaEventQuery(s.source_done);
  if(status==cudaErrorNotReady)return false;ck(status);return true;}
 std::vector<Result> poll()override{
  std::vector<Result> out;for(auto&p:slots){auto&s=*p;if(!s.busy||!s.recorded)continue;
   auto status=cudaEventQuery(s.done);if(status==cudaErrorNotReady)continue;ck(status);
   uint64_t cpu_start=mono_ns();Result r;r.meta=std::move(s.meta);r.backend=cfg.backend;
   auto duration=[](cudaEvent_t a,cudaEvent_t b){float ms=0;ck(cudaEventElapsedTime(&ms,a,b));return double(ms);};
   r.device_ms=duration(s.start,s.done);r.copy_ms=duration(s.start,s.copy_end);r.preprocess_ms=duration(s.copy_end,s.prep_end);
   r.inference_ms=duration(s.infer_start,s.infer_end);r.post_ms=duration(s.infer_end,s.done);
   if(cfg.verify){for(size_t i=0;i<s.reference.size();i++){float actual=input_half?__half2float(static_cast<__half*>(s.check)[i]):static_cast<float*>(s.check)[i];r.preprocess_error=std::max(r.preprocess_error,double(std::abs(actual-s.reference[i])));}
    if(r.preprocess_error>0.002)throw std::runtime_error("CUDA preprocess differs from matched CPU reference");}
   if(cfg.backend=="trt"){
    for(int i=0;i<anchors;i++){
     float row[6];for(int j=0;j<6;j++)row[j]=output_half?__half2float(static_cast<__half*>(s.host_output)[i*6+j]):static_cast<float*>(s.host_output)[i*6+j];
     auto box=decode_detr_row(row,r.meta.w,r.meta.h,cfg.confidence);if(box)r.boxes.push_back(*box);
    } // Top-k already lives in the graph; DETR results are NOT passed through NMS.
   }else{auto*b=s.host_bounds;if(b[2]>=b[0])r.boxes.push_back({float(b[0]),float(b[1]),float(b[2]+1),float(b[3]+1),1,0});}
   if(s.external&&r.meta.snapshot_requested)r.meta.snapshot=thumbnail({s.host,r.meta.w,r.meta.h});
   r.cpu_ms=(mono_ns()-cpu_start)/1e6;r.latency_ms=(mono_ns()-r.meta.captured_mono)/1e6;
   out.push_back(std::move(r));s.busy=false;s.recorded=false;
  }return out;
 }
 bool busy()override{for(auto&s:slots)if(s->busy)return true;return false;}
};
std::unique_ptr<Backend>make_cuda(const Settings&s){return std::make_unique<Cuda>(s);}
}
