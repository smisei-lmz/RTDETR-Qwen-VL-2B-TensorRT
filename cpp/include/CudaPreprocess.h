#pragma once
#include <cuda_runtime_api.h>

// Reused YOLO color/layout/normalization kernel. Resize is performed by OpenCV
// on uint8 first, matching the Python RT-DETR preprocessing exactly.
class CudaPreprocessor {
public:
    CudaPreprocessor() = default;
    ~CudaPreprocessor();
    CudaPreprocessor(const CudaPreprocessor&) = delete;
    CudaPreprocessor& operator=(const CudaPreprocessor&) = delete;
    cudaError_t CudaBGR2RGB_HWC2CHW_Normalize(
        const unsigned char* input, float* output, int width, int height,
        cudaStream_t stream = nullptr);
    // Build fixed-point resize coefficients once for this source resolution.
    cudaError_t configureResize(int src_width, int src_height, int dst_width, int dst_height);
    cudaError_t resizeNormalize(const unsigned char* input, int src_stride,
                               float* output, cudaStream_t stream);
private:
    void* x_map_{nullptr};
    void* y_map_{nullptr};
    int src_width_{0}, src_height_{0}, dst_width_{0}, dst_height_{0};
};
