#include "edge.hpp"
#include <csignal>
#include <filesystem>
#include <iostream>
#include <thread>
using namespace edge;
volatile std::sig_atomic_t interrupted=0;
void signal_handler(int){interrupted=1;}
struct Item {Lease lease;Meta meta;};
int main(int argc,char**argv){
 try {
    Settings settings;int frames=300,pool_size=8,queue_size=2;double fps=30,snapshot_hz=0;
    std::string source="synthetic",summary,trace,fault;bool serial=false;
    auto next=[&](int&i){if(i+1>=argc)throw std::runtime_error("missing argument");return std::string(argv[++i]);};
    for(int i=1;i<argc;i++){std::string a=argv[i];
      if(a=="--backend")settings.backend=next(i);else if(a=="--engine")settings.engine=next(i);
      else if(a=="--source")source=next(i);else if(a=="--frames")frames=std::stoi(next(i));
      else if(a=="--fps")fps=std::stod(next(i));else if(a=="--pool")pool_size=std::stoi(next(i));
      else if(a=="--queue")queue_size=std::stoi(next(i));else if(a=="--lanes")settings.lanes=std::stoi(next(i));
      else if(a=="--work-ms")settings.work_ms=std::stod(next(i));else if(a=="--snapshot-hz")snapshot_hz=std::stod(next(i));
      else if(a=="--max-pixels")settings.max_pixels=std::stoi(next(i));else if(a=="--summary")summary=next(i);
      else if(a=="--trace")trace=next(i);else if(a=="--fault")fault=next(i);
      else if(a=="--verify-preprocess")settings.verify=true;else if(a=="--serial")serial=true;
      else if(a=="--help"){std::cout<<"edge_pipeline --backend cpu|cuda|trt --source synthetic|PPM-directory --frames N --fps N --lanes N --pool N --queue N --engine FILE --verify-preprocess --serial --snapshot-hz N --trace FILE --summary FILE\n";return 0;}
      else throw std::runtime_error("unknown argument: "+a);
    }
    if(frames<1||fps<=0||settings.lanes<1||pool_size<1||queue_size<1||settings.work_ms<0||snapshot_hz<0)throw std::runtime_error("invalid positive count/rate");
    if(serial)settings.lanes=1;
    std::vector<std::string> paths;if(source!="synthetic"){
      for(auto&p:std::filesystem::directory_iterator(source))if(p.path().extension()==".ppm")paths.push_back(p.path().string());
      std::sort(paths.begin(),paths.end());if(paths.empty())throw std::runtime_error("source has no .ppm files");}
    std::unique_ptr<Backend> backend;
    if(settings.backend=="cpu")backend=make_cpu(settings);
    else {
#ifdef EDGE_CUDA
      backend=make_cuda(settings);
#else
      throw std::runtime_error("CUDA backend not built; configure -DEDGE_WITH_CUDA=ON");
#endif
    }
    Pool pool(pool_size);Queue<Item> ready(queue_size);std::atomic<bool>stop{false},capture_done{false};
    std::atomic<uint64_t>generated{0},drop_old{0},drop_pool{0};uint64_t completed=0;std::vector<double>latency;
    std::exception_ptr producer_error;std::string session=std::to_string(unix_ns());
    std::ofstream trace_file;if(!trace.empty()){trace_file.open(trace);if(!trace_file)throw std::runtime_error("cannot write trace");}
    std::signal(SIGINT,signal_handler);auto start=mono_ns();uint64_t last_snapshot=0;
    auto make_item=[&](int frame)->std::optional<Item>{
      generated++;auto lease=pool.acquire();if(!lease){drop_pool++;return {};}
      try{auto& im=pool.writable(*lease);if(source=="synthetic")synthetic_into(im,frame);else read_ppm_into(paths[frame%paths.size()],im);
        Meta meta;meta.frame=frame;meta.session=session;meta.captured_mono=mono_ns();meta.captured_unix=unix_ns();
        meta.slot=lease->slot;meta.generation=lease->generation;meta.w=im.w;meta.h=im.h;
        pool.publish(*lease);return Item{*lease,std::move(meta)};
      }catch(...){pool.release(*lease);throw;}
    };
    auto consume=[&](){for(auto&r:backend->poll()){
      completed++;latency.push_back(r.latency_ms);std::string line=json(r);std::cout<<line<<'\n';
      if(snapshot_hz>0)std::cout.flush();if(trace_file)trace_file<<line<<'\n';
    }};
    auto submit=[&](Item item,int lane){bool owned=true;try{
      if(fault=="early-recycle"){pool.release(item.lease);owned=false;}
      View view=pool.read(item.lease);uint64_t now=mono_ns();
      if(snapshot_hz>0&&now-last_snapshot>=uint64_t(1e9/snapshot_hz)){item.meta.snapshot=thumbnail(view);last_snapshot=now;}
      item.meta.submitted=mono_ns();backend->submit(lane,view,std::move(item.meta));
      // CPU source can be returned: backend copied it to its own pinned storage, or finished CPU reads.
      pool.release(item.lease);owned=false;
    }catch(...){if(owned)pool.release(item.lease);throw;}};
    if(serial){for(int f=0;f<frames&&!interrupted;f++){auto item=make_item(f);if(item)submit(std::move(*item),0);
        while(backend->busy()){consume();std::this_thread::sleep_for(std::chrono::milliseconds(1));}}
      capture_done=true;
    }else{
      std::thread capture([&]{try{auto next=std::chrono::steady_clock::now();
        for(int f=0;f<frames&&!stop&&!interrupted;f++){
          auto item=make_item(f);if(item){auto dropped=ready.push(std::move(*item));if(dropped){pool.release(dropped->lease);drop_old++;}}
          next+=std::chrono::microseconds(int64_t(1e6/fps));
          while(!stop&&!interrupted&&std::chrono::steady_clock::now()<next)std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
      }catch(...){producer_error=std::current_exception();}capture_done=true;ready.close();});
      try{
        while(!capture_done||!ready.empty()||backend->busy()){
          if(interrupted)stop=true;consume();int lane=backend->available();
          if(lane>=0){auto item=ready.pop();if(item)submit(std::move(*item),lane);}
          else std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
      }catch(...){stop=true;capture.join();while(auto item=ready.pop())pool.release(item->lease);throw;}
      capture.join();if(producer_error)std::rethrow_exception(producer_error);
    }
    if(!pool.all_free()||generated!=completed+drop_old+drop_pool)throw std::runtime_error("ownership/accounting invariant failed");
    std::sort(latency.begin(),latency.end());auto p=[&](double n){return latency.empty()?0:latency[size_t(n*(latency.size()-1))];};
    std::ostringstream s;s<<"{\"generated\":"<<generated<<",\"completed\":"<<completed<<",\"drop_oldest\":"<<drop_old
      <<",\"drop_no_slot\":"<<drop_pool<<",\"queue_highwater\":"<<ready.highwater<<",\"all_slots_free\":true,\"elapsed_s\":"<<(mono_ns()-start)/1e9
      <<",\"latency_p50_ms\":"<<p(.5)<<",\"latency_p95_ms\":"<<p(.95)<<",\"latency_p99_ms\":"<<p(.99)<<"}";
    std::cerr<<s.str()<<'\n';if(!summary.empty()){std::ofstream f(summary);if(!f)throw std::runtime_error("cannot write summary");f<<s.str()<<'\n';}
    return 0;
 }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}
}
