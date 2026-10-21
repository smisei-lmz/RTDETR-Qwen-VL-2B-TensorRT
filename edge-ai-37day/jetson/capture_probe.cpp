#include "v4l2_camera.hpp"
#include <iostream>
#include <thread>
#include <csignal>
using namespace edge;
volatile std::sig_atomic_t stopped=0;
int main(int argc,char**argv){try{
 std::string device="/dev/video0",format="YUYV";int frames=120,w=640,h=480,hold=0;bool export_fd=false;
 for(int i=1;i<argc;i++){std::string a=argv[i];auto next=[&]{if(++i>=argc)throw std::runtime_error("missing argument");return std::string(argv[i]);};
 if(a=="--camera")device=next();else if(a=="--format")format=next();else if(a=="--frames")frames=std::stoi(next());
 else if(a=="--width")w=std::stoi(next());else if(a=="--height")h=std::stoi(next());else if(a=="--hold-ms")hold=std::stoi(next());else if(a=="--export")export_fd=true;else throw std::runtime_error("unknown argument");}
 if(format!="YUYV"&&format!="NV12")throw std::runtime_error("format must be YUYV or NV12");
 Camera camera(device,w,h,format=="YUYV"?V4L2_PIX_FMT_YUYV:V4L2_PIX_FMT_NV12);int n=camera.allocate(6,false,export_fd);
 for(int i=0;i<n;i++)camera.queue(i);camera.start();std::signal(SIGINT,[](int){stopped=1;});int count=0,timeouts=0;
 while(count<frames&&!stopped){auto frame=camera.next(1000);if(!frame){if(++timeouts>5)throw std::runtime_error("capture stalled");continue;}timeouts=0;
 std::cout<<"{\"index\":"<<frame->index<<",\"generation\":"<<frame->generation<<",\"sequence\":"<<frame->sequence
   <<",\"driver_timestamp_ns\":"<<frame->driver_ns<<",\"flags\":"<<frame->flags<<",\"bytesused\":"<<frame->bytes
   <<",\"valid\":"<<(camera.valid(*frame)?"true":"false")<<",\"export_fd\":"<<camera.exported_fd(frame->index)<<"}\n";
 if(hold>0)std::this_thread::sleep_for(std::chrono::milliseconds(hold));camera.release(*frame);count++;}
 camera.stop();return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
