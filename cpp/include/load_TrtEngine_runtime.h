#pragma once 

#include <Nvinfer.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <memory>
#include "Logger.h"
#include "BufferManager.h"
#include <vector>

class TrtEngine{
public:
    TrtEngine();
    ~TrtEngine();
    bool loadEngine(const std::string path);
    bool inputSize(){
        return input_size_;
    }
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


private:
    int input_size_{0};
    int output_size_{0};
    int input_width_{0};
    int input_height_{0};
    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
};