#include "edge.hpp"
namespace edge {
class Cpu:public Backend {
    Settings cfg;std::vector<std::optional<Result>> jobs;std::vector<uint64_t> ready;
public:
    explicit Cpu(Settings s):cfg(s),jobs(s.lanes),ready(s.lanes){}
    int available()override{for(size_t i=0;i<jobs.size();i++)if(!jobs[i])return int(i);return -1;}
    void submit(int lane,View src,Meta meta)override{
        if(jobs.at(lane))throw std::runtime_error("lane still in use");uint64_t start=mono_ns();
        Result r;r.meta=std::move(meta);r.backend="cpu";r.boxes=color_detect(src);r.cpu_ms=(mono_ns()-start)/1e6;
        ready[lane]=mono_ns()+uint64_t(cfg.work_ms*1e6);jobs[lane]=std::move(r);
    }
    std::vector<Result> poll()override{std::vector<Result>out;for(size_t i=0;i<jobs.size();i++)if(jobs[i]&&mono_ns()>=ready[i]){
      jobs[i]->latency_ms=(mono_ns()-jobs[i]->meta.captured_mono)/1e6;out.push_back(std::move(*jobs[i]));jobs[i].reset();}return out;}
    bool busy()override{for(auto&j:jobs)if(j)return true;return false;}
};
std::unique_ptr<Backend> make_cpu(const Settings&s){return std::make_unique<Cpu>(s);}
}
