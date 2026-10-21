#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace edge {
inline uint64_t mono_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count(); }
inline uint64_t unix_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count(); }
struct Image { int w=0,h=0; std::vector<uint8_t> rgb; };
struct View { const uint8_t* rgb; int w,h; };
struct Box { float x1,y1,x2,y2,score; int cls; };
struct Meta {
    uint64_t frame=0,generation=0,captured_mono=0,captured_unix=0,submitted=0;
    uint64_t driver_timestamp_ns=0;uint32_t driver_flags=0;
    int slot=-1,w=0,h=0; bool snapshot_requested=false; std::string session,snapshot;
};
struct Result { Meta meta; std::vector<Box> boxes;std::string backend;
 double latency_ms=0,device_ms=0,preprocess_error=0,copy_ms=0,preprocess_ms=0,inference_ms=0,post_ms=0,cpu_ms=0; };
inline std::string quote(const std::string& text) {
    std::ostringstream out; out << '"';
    for(unsigned char c:text) { if(c=='"'||c=='\\') out << '\\' << c;
        else if(c=='\n') out << "\\n"; else if(c=='\r') out << "\\r";
        else if(c<32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else out << c; }
    out << '"'; return out.str();
}
inline std::string json(const Result& r) {
    std::ostringstream o; o << std::setprecision(9) << "{\"type\":\"result\",\"session_id\":" << quote(r.meta.session)
      << ",\"backend\":" << quote(r.backend)
      << ",\"frame_id\":" << r.meta.frame << ",\"slot\":" << r.meta.slot << ",\"generation\":" << r.meta.generation
      << ",\"captured_unix_ns\":" << r.meta.captured_unix << ",\"captured_mono_ns\":" << r.meta.captured_mono
      << ",\"driver_timestamp_ns\":" << r.meta.driver_timestamp_ns << ",\"driver_flags\":" << r.meta.driver_flags
      << ",\"width\":" << r.meta.w << ",\"height\":" << r.meta.h << ",\"latency_ms\":" << r.latency_ms
      << ",\"device_ms\":" << r.device_ms << ",\"preprocess_max_error\":" << r.preprocess_error
      << ",\"copy_or_yuv_ms\":" << r.copy_ms << ",\"preprocess_ms\":" << r.preprocess_ms
      << ",\"inference_ms\":" << r.inference_ms << ",\"gpu_post_d2h_ms\":" << r.post_ms << ",\"cpu_ms\":" << r.cpu_ms
      << ",\"snapshot_ppm\":" << quote(r.meta.snapshot) << ",\"boxes\":[";
    for(size_t i=0;i<r.boxes.size();++i) {auto b=r.boxes[i]; if(i)o<<',';
      o << "{\"class_id\":" << b.cls << ",\"score\":" << b.score << ",\"xyxy\":["
        << b.x1 << ',' << b.y1 << ',' << b.x2 << ',' << b.y2 << "]}";}
    o << "]}"; return o.str();
}
inline std::string base64(const std::string& in) {
    constexpr char abc[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out; unsigned val=0; int bits=-6;
    for(unsigned char c:in) { val=(val<<8)+c; bits+=8; while(bits>=0){out.push_back(abc[(val>>bits)&63]);bits-=6;} }
    if(bits>-6)out.push_back(abc[((val<<8)>>(bits+8))&63]);
    while(out.size()%4)out.push_back('='); return out;
}
inline std::string thumbnail(View src, int width=640) {
    int w=std::min(width,src.w),h=std::max(1,src.h*w/src.w);
    std::string p="P6\n"+std::to_string(w)+" "+std::to_string(h)+"\n255\n";
    for(int y=0;y<h;y++)for(int x=0;x<w;x++) {
        size_t i=(size_t(y*src.h/h)*src.w+x*src.w/w)*3;
        p.append(reinterpret_cast<const char*>(src.rgb+i),3);
    }
    return base64(p);
}
inline void synthetic_into(Image& im,uint64_t id,int w=640,int h=360) {
    im.w=w;im.h=h;im.rgb.resize(size_t(w)*h*3);std::fill(im.rgb.begin(),im.rgb.end(),0);
    int bw=std::max(1,w/5),bh=std::max(1,h/4),left=int((id*7)%std::max(1,w-bw)),top=h/3;
    for(int y=top;y<std::min(h,top+bh);y++)for(int x=left;x<left+bw;x++)im.rgb[(size_t(y)*w+x)*3]=255;
}
inline Image synthetic(uint64_t id,int w=640,int h=360){Image image;synthetic_into(image,id,w,h);return image;}
inline void read_ppm_into(const std::string& path,Image& im) {
    std::ifstream f(path,std::ios::binary); if(!f)throw std::runtime_error("cannot open PPM: "+path);
    auto token=[&](){std::string t; while(f>>t){if(t[0]=='#'){f.ignore(std::numeric_limits<std::streamsize>::max(),'\n');continue;}return t;}throw std::runtime_error("bad PPM header");};
    if(token()!="P6")throw std::runtime_error("only binary P6 PPM supported");
    int w=std::stoi(token()),h=std::stoi(token()),v=std::stoi(token());
    if(w<1||h<1||w>4096||h>4096||v!=255)throw std::runtime_error("invalid PPM dimensions/range");
    char sep=0;f.get(sep);if(sep=='\r'&&f.peek()=='\n')f.get();
    im.w=w;im.h=h;im.rgb.resize(size_t(w)*h*3);
    f.read(reinterpret_cast<char*>(im.rgb.data()),im.rgb.size());if(!f)throw std::runtime_error("short PPM data");
}
inline Image read_ppm(const std::string& path){Image image;read_ppm_into(path,image);return image;}
inline std::vector<Box> color_detect(View v) {
    int x0=v.w,y0=v.h,x1=-1,y1=-1;
    for(int y=0;y<v.h;y++)for(int x=0;x<v.w;x++){auto p=v.rgb+(size_t(y)*v.w+x)*3;
      if(p[0]>160&&p[1]<100&&p[2]<100){x0=std::min(x0,x);y0=std::min(y0,y);x1=std::max(x1,x);y1=std::max(y1,y);}}
    if(x1<x0)return {};return {{float(x0),float(y0),float(x1+1),float(y1+1),1.f,0}};
}
inline std::vector<float> preprocess_reference(View s,int w,int h) {
    // RT-DETR: stretch resize, half-pixel bilinear, RGB /255. No padding.
    std::vector<float> out(size_t(w)*h*3);
    for(int y=0;y<h;y++)for(int x=0;x<w;x++){
      float fx=std::max(0.f,std::min(float(s.w-1),(x+.5f)*s.w/w-.5f));
      float fy=std::max(0.f,std::min(float(s.h-1),(y+.5f)*s.h/h-.5f));
      int x0=int(fx),y0=int(fy),x1=std::min(x0+1,s.w-1),y1=std::min(y0+1,s.h-1);float ax=fx-x0,ay=fy-y0;
      for(int c=0;c<3;c++){
        auto get=[&](int xx,int yy){return float(s.rgb[(size_t(yy)*s.w+xx)*3+c]);};
        float z=(1-ay)*((1-ax)*get(x0,y0)+ax*get(x1,y0))+ay*((1-ax)*get(x0,y1)+ax*get(x1,y1));
        out[(size_t(c)*h+y)*w+x]=z/255;
      }
    }return out;
}
inline std::optional<Box> decode_detr_row(const float* row,int w,int h,float threshold=.25f) {
    for(int i=0;i<6;i++)if(!std::isfinite(row[i]))throw std::runtime_error("non-finite DETR row");
    if(row[4]<0||row[4]>1||row[5]!=std::floor(row[5])||row[5]<0||row[5]>=80)
        throw std::runtime_error("invalid DETR score/class");
    auto clamp=[](float v){return std::max(0.f,std::min(1.f,v));};
    Box box{clamp(row[0])*w,clamp(row[1])*h,clamp(row[2])*w,clamp(row[3])*h,row[4],int(row[5])};
    if(box.score<threshold||box.x2<=box.x1||box.y2<=box.y1)return {};
    return box;
}
struct Lease {int slot;uint64_t generation;};
class Pool {
    enum State {FREE,FILLING,READY,READING};
    struct Slot {Image image;uint64_t generation=0;State state=FREE;};
    std::vector<Slot> slots; mutable std::mutex mutex;
    Slot& check(Lease l){auto&s=slots.at(l.slot);if(s.generation!=l.generation||s.state==FREE)throw std::runtime_error("stale/early-recycled lease");return s;}
public:
    explicit Pool(int n):slots(n){if(n<1)throw std::runtime_error("pool must be positive");}
    std::optional<Lease> acquire(){std::lock_guard<std::mutex> lock(mutex);for(size_t i=0;i<slots.size();i++)if(slots[i].state==FREE){auto&s=slots[i];s.state=FILLING;return Lease{int(i),++s.generation};}return {};}
    Image& writable(Lease l){std::lock_guard<std::mutex>lock(mutex);auto&s=check(l);if(s.state!=FILLING)throw std::runtime_error("not filling");return s.image;}
    void publish(Lease l){std::lock_guard<std::mutex>lock(mutex);auto&s=check(l);if(s.state!=FILLING)throw std::runtime_error("not filling");s.state=READY;}
    void fill(Lease l,Image image){std::lock_guard<std::mutex> lock(mutex);auto&s=check(l);if(s.state!=FILLING)throw std::runtime_error("not filling");s.image=std::move(image);s.state=READY;}
    View read(Lease l){std::lock_guard<std::mutex> lock(mutex);auto&s=check(l);if(s.state!=READY)throw std::runtime_error("not ready");s.state=READING;return {s.image.rgb.data(),s.image.w,s.image.h};}
    void release(Lease l){std::lock_guard<std::mutex> lock(mutex);check(l).state=FREE;}
    bool all_free()const{std::lock_guard<std::mutex>lock(mutex);for(auto&s:slots)if(s.state!=FREE)return false;return true;}
};
template<class T>class Queue {
    size_t cap;std::deque<T> q;mutable std::mutex mutex;std::condition_variable cv;bool closed=false;
public:
    size_t highwater=0;
    explicit Queue(size_t n):cap(n){if(!n)throw std::runtime_error("queue must be positive");}
    std::optional<T> push(T item){std::lock_guard<std::mutex>lock(mutex);if(closed)throw std::runtime_error("closed queue");std::optional<T> old;
      if(q.size()==cap){old=std::move(q.front());q.pop_front();}q.push_back(std::move(item));highwater=std::max(highwater,q.size());cv.notify_one();return old;}
    std::optional<T> pop(){std::unique_lock<std::mutex>lock(mutex);cv.wait_for(lock,std::chrono::milliseconds(1),[&]{return !q.empty()||closed;});
      if(q.empty())return {};T item=std::move(q.front());q.pop_front();return item;}
    bool empty(){std::lock_guard<std::mutex>lock(mutex);return q.empty();}
    void close(){std::lock_guard<std::mutex>lock(mutex);closed=true;cv.notify_all();}
};
struct Settings {
    std::string backend="cpu",engine;int lanes=2,max_pixels=1920*1080;double work_ms=0;
    float confidence=.25f;bool verify=false;
};
// Device pointers are obtained from a validated platform importer, never cast from an fd.
struct DeviceYuv {const uint8_t* y;const uint8_t* uv;int w,h,pitch_y,pitch_uv;int format;int matrix=601;bool full_range=false;uint64_t tex_y=0,tex_uv=0;};
class Backend {
public:
    virtual ~Backend()=default;
    virtual int available()=0;
    // submit must finish reading the CPU source (normally copy to its pinned slot) before returning.
    virtual void submit(int lane,View source,Meta meta)=0;
    virtual void submit_device(int,DeviceYuv,Meta){throw std::runtime_error("external YUV input is unavailable");}
    virtual bool source_ready(int){return true;}
    virtual std::vector<Result> poll()=0;
    virtual bool busy()=0;
};
std::unique_ptr<Backend> make_cpu(const Settings&);
#ifdef EDGE_CUDA
std::unique_ptr<Backend> make_cuda(const Settings&);
#endif
} // namespace edge
