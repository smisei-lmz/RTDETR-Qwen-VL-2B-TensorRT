#include "CudaPreprocess.h"
#include <cuda_runtime.h>

namespace {
__global__ void BGR2RGB_HWC2CHW_Normalize_kernel(
    const unsigned char* input, float* output, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int dst = y * width + x;
    const int src = dst * 3;
    const int area = width * height;
    output[dst] = input[src + 2] / 255.0f;
    output[area + dst] = input[src + 1] / 255.0f;
    output[2 * area + dst] = input[src] / 255.0f;
}
}

cudaError_t CudaPreprocessor::CudaBGR2RGB_HWC2CHW_Normalize(
    const unsigned char* input, float* output, int width, int height, cudaStream_t stream) {
    if (!input || !output || width <= 0 || height <= 0) return cudaErrorInvalidValue;
    const dim3 threads(16, 16);
    const dim3 blocks((width + 15) / 16, (height + 15) / 16);
    BGR2RGB_HWC2CHW_Normalize_kernel<<<blocks, threads, 0, stream>>>(input, output, width, height);
    return cudaGetLastError();
}

#include <algorithm>
#include <cmath>
#include <vector>

namespace {
struct ResizeAxis { int first, second, w0, w1; };

std::vector<ResizeAxis> makeAxis(int source, int target, bool horizontal) {
    std::vector<ResizeAxis> result(target);
    const double scale = static_cast<double>(source) / target;
    for (int i = 0; i < target; ++i) {
        float fraction = static_cast<float>((i + 0.5) * scale - 0.5);
        int first = static_cast<int>(std::floor(fraction));
        fraction -= first;
        // OpenCV's uint8 INTER_LINEAR uses 11-bit coefficients. Horizontal
        // edge coefficients collapse; vertical row selection clips separately.
        if (horizontal && (first < 0 || first >= source - 1)) {
            first = std::max(0, std::min(first, source - 1));
            fraction = 0;
        }
        result[i] = {std::max(0, std::min(first, source - 1)),
                     std::max(0, std::min(first + 1, source - 1)),
                     static_cast<int>(std::lrint((1.0f - fraction) * 2048)),
                     static_cast<int>(std::lrint(fraction * 2048))};
    }
    return result;
}

__global__ void resizeNormalizeKernel(const unsigned char* input, int stride,
    float* output, int width, int height, const ResizeAxis* xs, const ResizeAxis* ys) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const ResizeAxis ax = xs[x], ay = ys[y];
    const int area = width * height, dst = y * width + x;
    for (int c = 0; c < 3; ++c) {
        const int top = input[ay.first * stride + ax.first * 3 + c] * ax.w0
                      + input[ay.first * stride + ax.second * 3 + c] * ax.w1;
        const int bottom = input[ay.second * stride + ax.first * 3 + c] * ax.w0
                         + input[ay.second * stride + ax.second * 3 + c] * ax.w1;
        // Retain the uint8 resize rounding before floating-point normalization.
        const int value = ((((ay.w0 * (top >> 4)) >> 16)
                          + ((ay.w1 * (bottom >> 4)) >> 16) + 2) >> 2);
        output[(2 - c) * area + dst] = value / 255.0f;
    }
}
}

CudaPreprocessor::~CudaPreprocessor() {
    if (x_map_) cudaFree(x_map_);
    if (y_map_) cudaFree(y_map_);
}

cudaError_t CudaPreprocessor::configureResize(int sw, int sh, int dw, int dh) {
    if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return cudaErrorInvalidValue;
    if (sw == src_width_ && sh == src_height_ && dw == dst_width_ && dh == dst_height_) return cudaSuccess;
    if (x_map_) cudaFree(x_map_);
    if (y_map_) cudaFree(y_map_);
    x_map_ = y_map_ = nullptr;
    src_width_ = src_height_ = dst_width_ = dst_height_ = 0;
    const auto xs = makeAxis(sw, dw, true), ys = makeAxis(sh, dh, false);
    auto error = cudaMalloc(&x_map_, xs.size() * sizeof(ResizeAxis));
    if (error != cudaSuccess) return error;
    error = cudaMalloc(&y_map_, ys.size() * sizeof(ResizeAxis));
    if (error != cudaSuccess) return error;
    error = cudaMemcpy(x_map_, xs.data(), xs.size() * sizeof(ResizeAxis), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return error;
    error = cudaMemcpy(y_map_, ys.data(), ys.size() * sizeof(ResizeAxis), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return error;
    src_width_ = sw; src_height_ = sh; dst_width_ = dw; dst_height_ = dh;
    return cudaSuccess;
}

cudaError_t CudaPreprocessor::resizeNormalize(const unsigned char* input, int stride,
    float* output, cudaStream_t stream) {
    if (!input || !output || !x_map_ || !y_map_ || stride < src_width_ * 3 || !dst_width_)
        return cudaErrorInvalidValue;
    const dim3 threads(16, 16), blocks((dst_width_ + 15) / 16, (dst_height_ + 15) / 16);
    resizeNormalizeKernel<<<blocks, threads, 0, stream>>>(input, stride, output,
        dst_width_, dst_height_, static_cast<const ResizeAxis*>(x_map_), static_cast<const ResizeAxis*>(y_map_));
    return cudaGetLastError();
}
