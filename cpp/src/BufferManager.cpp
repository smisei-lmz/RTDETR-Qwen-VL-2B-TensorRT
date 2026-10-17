#include "BufferManager.h"

#include <stdexcept>
#include <string>

BufferManager::BufferManager(){
    input_ = nullptr;
    output_ = nullptr;
}


BufferManager::~BufferManager(){
    release();
}

void BufferManager::allocate(std::size_t inputBytes, std::size_t outputBytes){
    release();
    const auto input_status = cudaMalloc(&input_,inputBytes);
    const auto output_status = input_status == cudaSuccess
        ? cudaMalloc(&output_,outputBytes) : input_status;
    if (output_status != cudaSuccess) {
        release();
        throw std::runtime_error(std::string("GPU buffer allocation failed: ") + cudaGetErrorString(output_status));
    }
    std::cout << "GPU buffer allocated"<< std::endl;
}

bool BufferManager::ensureRawCapacity(std::size_t bytes) {
    if (bytes == 0) {
        return false;
    }

    if (raw_ != nullptr && bytes <= raw_capacity_) {
        return true;
    }

    if (raw_ != nullptr) {
        cudaFree(raw_);
        raw_ = nullptr;
        raw_capacity_ = 0;
    }

    cudaError_t error = cudaMalloc(&raw_, bytes);

    if (error != cudaSuccess) {
        raw_ = nullptr;
        return false;
    }

    raw_capacity_ = bytes;
    return true;
}

void BufferManager::release() noexcept{
    if (input_){
        cudaFree(input_);
        input_ = nullptr;
    }
    if (output_){
        cudaFree(output_);
        output_ = nullptr;
    }
    if (raw_) {
        cudaFree(raw_);
        raw_ = nullptr;
        raw_capacity_ = 0;
    }
}
