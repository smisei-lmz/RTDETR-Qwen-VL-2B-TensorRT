#include "CudaPostprocess.h"
#include <cuda_runtime.h>

namespace {
__global__ void decodeKernel(const float* output, int n, float threshold,
    int width, int height, int classes, Detection* detections, int* count) {
    extern __shared__ int scan[];
    const int i = threadIdx.x;
    Detection det{};
    bool valid = i < n;
    if (valid) {
        const float* row = output + i * 6;
        for (int j = 0; j < 6; ++j) valid &= isfinite(row[j]);
        valid = valid && row[4] >= threshold && row[2] > 0 && row[3] > 0
                      && row[5] >= 0 && row[5] < classes;
        if (valid) {
            det.class_id = static_cast<int>(roundf(row[5]));
            valid = det.class_id < classes && fabsf(row[5] - det.class_id) <= 1e-3f;
            det.confidence = row[4];
            det.x1 = (row[0] - row[2] * 0.5f) * width;
            det.y1 = (row[1] - row[3] * 0.5f) * height;
            det.x2 = (row[0] + row[2] * 0.5f) * width;
            det.y2 = (row[1] + row[3] * 0.5f) * height;
        }
    }
    scan[i] = valid ? 1 : 0;
    __syncthreads();
    // A single-block inclusive scan keeps the original Top-K prediction order.
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        const int previous = i >= offset ? scan[i - offset] : 0;
        __syncthreads();
        scan[i] += previous;
        __syncthreads();
    }
    if (valid) detections[scan[i] - 1] = det;
    if (i == blockDim.x - 1) *count = scan[i];
}
}

cudaError_t CudaPostprocessor::decode(const float* output, int n, float threshold,
    int width, int height, int classes, Detection* detections, int* count, cudaStream_t stream) {
    if (!output || !detections || !count || n <= 0 || n > 1024 || width <= 0 || height <= 0 || classes <= 0)
        return cudaErrorInvalidValue;
    int threads = 1;
    while (threads < n) threads <<= 1;
    decodeKernel<<<1, threads, threads * sizeof(int), stream>>>(output, n, threshold,
        width, height, classes, detections, count);
    return cudaGetLastError();
}
