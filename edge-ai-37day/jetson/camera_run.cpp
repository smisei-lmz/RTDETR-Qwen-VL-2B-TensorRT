#include "v4l2_camera.hpp"
#ifdef EDGE_JETSON
#include "nv_pool.hpp"
#endif
#include <iostream>
#include <thread>
#include <csignal>
#include <map>
using namespace edge;
volatile std::sig_atomic_t stopped=0;
struct StopCamera {Camera&camera;~StopCamera(){camera.stop();}};
int main(int argc,char**argv){try{
 Settings cfg;cfg.backend="cuda";std::string device="/dev/video0",format="YUYV",mode="copy";
 int w=640,h=480,frames=300,matrix=601;bool full=false,verify_source=false;double snapshot_hz=0,duration=0;
 for(int i=1;i<argc;i++){std::string a=argv[i];auto next=[&]{if(++i>=argc)throw std::runtime_error("missing argument");return std::string(argv[i]);};
  if(a=="--camera")device=next();else if(a=="--format")format=next();else if(a=="--mode")mode=next();else if(a=="--engine")cfg.engine=next();
  else if(a=="--backend")cfg.backend=next();else if(a=="--width")w=std::stoi(next());else if(a=="--height")h=std::stoi(next());
  else if(a=="--frames")frames=std::stoi(next());else if(a=="--matrix")matrix=std::stoi(next());else if(a=="--full-range")full=true;
  else if(a=="--snapshot-hz")snapshot_hz=std::stod(next());else if(a=="--duration-s")duration=std::stod(next());else if(a=="--verify-source")verify_source=true;else throw std::runtime_error("unknown option "+a);
 }
 if((format!="NV12"&&format!="YUYV")||(mode!="copy"&&mode!="shared")||(matrix!=601&&matrix!=709)||frames<1||snapshot_hz<0)throw std::runtime_error("invalid camera arguments");
 if(verify_source&&(mode!="shared"||cfg.backend!="cuda"))throw std::runtime_error("--verify-source requires shared + cuda color detector");
 bool shared=mode=="shared";if(shared&&cfg.backend=="cpu")throw std::runtime_error("shared mode needs GPU");
 Camera camera(device,w,h,format=="YUYV"?V4L2_PIX_FMT_YUYV:V4L2_PIX_FMT_NV12);
 cfg.max_pixels=std::max(cfg.max_pixels,int(camera.format.width*camera.format.height));int count=camera.allocate(6,shared,false);
#ifdef EDGE_JETSON
 std::unique_ptr<NvPool> surfaces;if(shared)surfaces=std::make_unique<NvPool>(count,camera.format);
#else
 if(shared)throw std::runtime_error("build with EDGE_WITH_JETSON=ON");
#endif
 std::unique_ptr<Backend> backend;if(cfg.backend=="cpu")backend=make_cpu(cfg);
 else{
#ifdef EDGE_CUDA
  backend=make_cuda(cfg);
#else
  throw std::runtime_error("CUDA not built");
#endif
 }
 StopCamera stop_guard{camera}; // stops DMA before backend/surface destruction on EVERY exit path.
 auto requeue=[&](const Captured&c){
#ifdef EDGE_JETSON
  if(shared){camera.release(c,surfaces->fd(c.index),surfaces->bytes(c.index));return;}
#endif
  camera.release(c);
 };
 for(int i=0;i<count;i++){
#ifdef EDGE_JETSON
  if(shared){camera.queue(i,surfaces->fd(i),surfaces->bytes(i));continue;}
#endif
  camera.queue(i);
 }
 camera.start();std::signal(SIGINT,[](int){stopped=1;});
 std::vector<std::optional<Captured>> held(cfg.lanes);std::map<uint64_t,std::vector<Box>>reference;
 uint64_t dequeued=0,completed=0,bad=0,last_frame=mono_ns(),last_snapshot=0,start=mono_ns();std::string session=std::to_string(unix_ns());
 auto any_held=[&]{for(auto&h:held)if(h)return true;return false;};
 auto capturing=[&]{return dequeued<uint64_t(frames)&&!stopped&&(duration<=0||(mono_ns()-start)/1e9<duration);};
 while(capturing()||backend->busy()||any_held()){
  // Query source completion BEFORE result polling clears a lane's event generation.
  for(int lane=0;lane<cfg.lanes;lane++)if(held[lane]&&backend->source_ready(lane)){requeue(*held[lane]);held[lane].reset();}
  for(auto&r:backend->poll()){
   if(verify_source){auto expected=reference.at(r.meta.frame);reference.erase(r.meta.frame);
    if(expected.size()!=r.boxes.size()||(!expected.empty()&&iou(expected[0],r.boxes[0])<.98f))throw std::runtime_error("shared-source CPU/GPU color-bounds mismatch; use a stable red target and inspect YUV colorimetry");}
   std::cout<<json(r)<<'\n';std::cout.flush();completed++;
  }
  int lane=backend->available();if(lane>=0&&capturing()){
   auto c=camera.next(5);if(!c){if(mono_ns()-last_frame>5000000000ULL)throw std::runtime_error("camera stalled for five seconds");continue;}
   last_frame=mono_ns();uint64_t id=dequeued++;if(!camera.valid(*c)){bad++;requeue(*c);continue;}
   Meta meta;meta.frame=id;meta.session=session;meta.slot=c->index;meta.generation=c->generation;
   meta.w=camera.format.width;meta.h=camera.format.height;meta.captured_mono=mono_ns();meta.captured_unix=unix_ns();
   meta.driver_timestamp_ns=c->driver_ns;meta.driver_flags=c->flags;
   meta.snapshot_requested=snapshot_hz>0&&meta.captured_mono-last_snapshot>=uint64_t(1e9/snapshot_hz);
   if(meta.snapshot_requested)last_snapshot=meta.captured_mono;
   if(shared){
#ifdef EDGE_JETSON
    if(verify_source){auto image=surfaces->cpu_reference(c->index,matrix,full);reference[id]=color_detect({image.rgb.data(),image.w,image.h});
     if(reference[id].empty())throw std::runtime_error("source verification requires a visible red target; refusing empty/empty parity");}
    held[lane]=*c;backend->submit_device(lane,surfaces->view(c->index,matrix,full),std::move(meta));
#endif
   }else{
    auto image=cpu_yuv(camera.data(c->index),camera.format,matrix,full);requeue(*c);
    if(meta.snapshot_requested)meta.snapshot=thumbnail({image.rgb.data(),image.w,image.h});
    backend->submit(lane,{image.rgb.data(),image.w,image.h},std::move(meta));
   }
  }else std::this_thread::sleep_for(std::chrono::microseconds(200));
 }
 if(dequeued!=completed+bad||any_held())throw std::runtime_error("camera accounting mismatch");
 camera.stop();std::cerr<<"{\"mode\":"<<quote(mode)<<",\"dequeued\":"<<dequeued<<",\"completed\":"<<completed<<",\"bad_frames\":"<<bad<<",\"held_sources\":0}\n";
 return 0;
}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}}
