#pragma once
#include <cuda_runtime_api.h>
#include "Types.h"

class CudaPostprocessor {
public:
    // Stable compaction for the current [1,N,6] export; N must be <=1024.
    // Results/count remain on device until the caller's D2H copy.
    static cudaError_t decode(const float* output, int num_boxes, float threshold,
        int image_width, int image_height, int num_classes,
        Detection* detections, int* count, cudaStream_t stream);
};
