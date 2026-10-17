#pragma once
#include <NvInfer.h>

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <memory>
#include "Logger.h"
#include "BufferManager.h"
#include <vector>

struct InferenceTiming {
    float h2d_ms{0.0f};
    float execute_ms{0.0f};
    float d2h_ms{0.0f};
    float gpu_total_ms{0.0f};
    double sync_wait_ms{0.0};
};

class TrtEngine{
public:
    TrtEngine();
    ~TrtEngine();
    bool loadEngine(const std::string& path);
    bool infer(float* input, float* output, InferenceTiming* timing = nullptr);
    bool infer_gpubuffer(float* output, InferenceTiming* timing = nullptr); //直接用gpu内存输入作为模型输入
    // Enqueue only; the caller orders preprocessing/copies and waits for completion.
    bool enqueue();
    int inputSize(){
        return input_size_;
    }
    int outputSize(){
        return output_size_;
    }
    int inputHeight(){
        return input_height_;
    }
    int inputWidth(){
        return input_width_;
    }
    BufferManager& buffers_info() noexcept{
        return buffers_;
    }
    cudaStream_t stream() const noexcept { return stream_; }
    
private:
    int input_size_{0};
    int output_size_{0};
    int input_width_{0};
    int input_height_{0};
    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    BufferManager buffers_;
    cudaStream_t stream_{};
    cudaEvent_t start_event_{};
    cudaEvent_t h2d_event_{};
    cudaEvent_t execute_event_{};
    cudaEvent_t d2h_event_{};
    std::string input_name_;
    std::string output_name_;
private:
    std::vector<char> loadFile(const std::string& path);
};
