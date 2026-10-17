#include "TrtEngine.h"
#include "nvtx_range.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>


using namespace nvinfer1;

bool TrtEngine::enqueue() {
    return context_ && context_->enqueueV3(stream_);
}

TrtEngine::TrtEngine(){
    if (cudaStreamCreate(&stream_) != cudaSuccess)
        throw std::runtime_error("Cannot create CUDA stream");
    if (cudaEventCreate(&start_event_) != cudaSuccess
        || cudaEventCreate(&h2d_event_) != cudaSuccess
        || cudaEventCreate(&execute_event_) != cudaSuccess
        || cudaEventCreate(&d2h_event_) != cudaSuccess) {
        if (start_event_) cudaEventDestroy(start_event_);
        if (h2d_event_) cudaEventDestroy(h2d_event_);
        if (execute_event_) cudaEventDestroy(execute_event_);
        if (d2h_event_) cudaEventDestroy(d2h_event_);
        cudaStreamDestroy(stream_);
        throw std::runtime_error("Cannot create CUDA timing events");
    }
}
TrtEngine::~TrtEngine(){
    cudaStreamSynchronize(stream_);
    context_.reset();
    buffers_.release();
    engine_.reset();
    runtime_.reset();
    cudaEventDestroy(start_event_);
    cudaEventDestroy(h2d_event_);
    cudaEventDestroy(execute_event_);
    cudaEventDestroy(d2h_event_);
    cudaStreamDestroy(stream_);
}
std ::vector<char> TrtEngine::loadFile(const std::string& path){
    std::ifstream file(path,std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open engine: " + path);
    file.seekg(0,std::ios::end);
    const auto end = file.tellg();
    if (end <= 0) throw std::runtime_error("Engine file is empty: " + path);
    size_t size = static_cast<size_t>(end);
    file.seekg(0,std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(),size)) throw std::runtime_error("Cannot read engine: " + path);
    return buffer;
}
bool TrtEngine::loadEngine(const std::string& path){
    auto data = loadFile(path);
    runtime_.reset(createInferRuntime(logger_));
    if (!runtime_) return false;
    engine_.reset(runtime_->deserializeCudaEngine(data.data(),data.size()));
    if(!engine_){
        std::cout<<"engine load failed"<<std::endl;
        return false;
    }
    context_.reset(engine_->createExecutionContext());
    if (!context_){
        std::cout << "context creation failed" << std::endl;
        return false;
    }
    int nb = engine_ -> getNbIOTensors();
    std::cout<<"IO number:"<<nb<<std::endl;
    if (nb != 2) {
        std::cerr << "Expected exactly one input and one output\n";
        return false;
    }
    for(int i=0;i<nb;i++){
        const char* name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        if (mode == TensorIOMode::kINPUT){
            input_name_ = name;
        }else{
            output_name_ = name;
        }
        auto shape = engine_->getTensorShape(name);
        if (engine_->getTensorDataType(name) != DataType::kFLOAT
            || engine_->getTensorFormat(name) != TensorFormat::kLINEAR
            || engine_->getTensorLocation(name) != TensorLocation::kDEVICE) {
            std::cerr << "Expected FP32 linear device I/O: " << name << '\n';
            return false;
        }
        if ((mode == TensorIOMode::kINPUT &&
             (shape.nbDims != 4 || shape.d[0] != 1 || shape.d[1] != 3)) ||
            (mode == TensorIOMode::kOUTPUT &&
             (shape.nbDims != 3 || shape.d[0] != 1 || shape.d[2] != 6))) {
            std::cerr << "Expected static [1,3,H,W] input and [1,N,6] RT-DETR output\n";
            return false;
        }
        int64_t size = 1;
        std::cout << "Tensor: " << name << " FP32 [";
        for (int j=0;j<shape.nbDims;j++){
            std::cout << (j ? "," : "") << shape.d[j];
            if (shape.d[j] <= 0 || size > std::numeric_limits<int>::max() / shape.d[j]) {
                std::cerr << "Unsupported dynamic or oversized tensor\n";
                return false;
            }
            size *= shape.d[j];
        }
        std::cout << "]\n";
        if (mode == TensorIOMode::kINPUT) {
            input_height_ = shape.d[2];  // H
            input_width_  = shape.d[3];  // W
        }
        if (mode == TensorIOMode::kINPUT){
            input_size_ = size;
        }else{
            output_size_ = size;
        }
        std::cout << "Tensor:" << name <<"size:" << size <<std::endl;
    }

    if (input_name_.empty() || output_name_.empty() || input_size_ <= 0 || output_size_ <= 0){
        std::cout << "invalid input/output tensors" << std::endl;
        return false;
    }

    buffers_.allocate(input_size_*sizeof(float),output_size_*sizeof(float));
    if (!context_->setTensorAddress(input_name_.c_str(),buffers_.input_device())
        || !context_->setTensorAddress(output_name_.c_str(),buffers_.output_device())){
        std::cout << "failed to bind tensor addresses" << std::endl;
        return false;
    }
    return true;
}


bool TrtEngine::infer(float* input, float* output, InferenceTiming* timing){
    if (!input || !output || !context_) return false;
    NvtxRange infer_range("infer");
    if (timing){
        cudaEventRecord(start_event_, stream_);
    }
    //H2D
    {
        NvtxRange range("input_h2d_enqueue");
        if (cudaMemcpyAsync(buffers_.input_device(),input,input_size_*sizeof(float),cudaMemcpyHostToDevice,stream_) != cudaSuccess)
            return false;
    }
    if (timing){
        cudaEventRecord(h2d_event_, stream_);
    }
    //execute
    bool success;
    {
        NvtxRange range("tensorrt_enqueue");
        success = context_->enqueueV3(stream_);
    }
    if (!success){
        std::cout << "TensorRT inference failed" <<std::endl;
        return false;
    }
    if (timing){
        cudaEventRecord(execute_event_, stream_);
    }
    //D2H
    {
        NvtxRange range("output_d2h_enqueue");
        if (cudaMemcpyAsync(output,buffers_.output_device(),output_size_*sizeof(float),cudaMemcpyDeviceToHost,stream_) != cudaSuccess)
            return false;
    }
    if (timing){
        cudaEventRecord(d2h_event_, stream_);
    }

    const auto sync_start = std::chrono::steady_clock::now();
    cudaError_t sync_status;
    {
        NvtxRange range("stream_sync");
        sync_status = cudaStreamSynchronize(stream_);
    }
    const auto sync_end = std::chrono::steady_clock::now();
    if (sync_status != cudaSuccess){
        std::cout << "CUDA stream synchronization failed: "
                  << cudaGetErrorString(sync_status) << std::endl;
        return false;
    }

    if (timing){
        cudaEventElapsedTime(&timing->h2d_ms, start_event_, h2d_event_);
        cudaEventElapsedTime(&timing->execute_ms, h2d_event_, execute_event_);
        cudaEventElapsedTime(&timing->d2h_ms, execute_event_, d2h_event_);
        cudaEventElapsedTime(&timing->gpu_total_ms, start_event_, d2h_event_);
        timing->sync_wait_ms = std::chrono::duration<double, std::milli>(
            sync_end - sync_start).count();
    }
    return true;
}


bool TrtEngine::infer_gpubuffer(float* output, InferenceTiming* timing){
    NvtxRange infer_range("infer_gpubuffer");
    if (output == nullptr || !context_) {
        return false;
    }
    if (timing){
        timing->h2d_ms = 0.0f;
        cudaEventRecord(start_event_,stream_);
    }
    // buffers_.input_device() 已经被 CUDA preprocess 填好了
        //execute
    bool success;
    {
        NvtxRange range("tensorrt_enqueue");
        success = context_->enqueueV3(stream_);
    }
    if (!success){
        std::cout << "TensorRT inference failed" <<std::endl;
        return false;
    }
    if (timing){
        cudaEventRecord(execute_event_, stream_);
    }
    //D2H
    {
        NvtxRange range("output_d2h_enqueue");
        if (cudaMemcpyAsync(output,buffers_.output_device(),output_size_*sizeof(float),cudaMemcpyDeviceToHost,stream_) != cudaSuccess)
            return false;
    }
    if (timing){
        cudaEventRecord(d2h_event_, stream_);
    }

    const auto sync_start = std::chrono::steady_clock::now();
    cudaError_t sync_status;
    {
        NvtxRange range("stream_sync");
        sync_status = cudaStreamSynchronize(stream_);
    }
    const auto sync_end = std::chrono::steady_clock::now();
    if (sync_status != cudaSuccess){
        std::cout << "CUDA stream synchronization failed: "
                  << cudaGetErrorString(sync_status) << std::endl;
        return false;
    }
    if (timing){
        cudaEventElapsedTime(&timing->execute_ms, start_event_, execute_event_);
        cudaEventElapsedTime(&timing->d2h_ms,execute_event_,d2h_event_);
        cudaEventElapsedTime(&timing->gpu_total_ms, start_event_, d2h_event_);
        timing->sync_wait_ms = std::chrono::duration<double, std::milli>(
            sync_end - sync_start).count();
    }
    return true;

}







