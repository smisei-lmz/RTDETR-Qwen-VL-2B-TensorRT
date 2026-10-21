#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/core/mat.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "BufferManager.h"
#include "CudaPreprocess.h"
#include "Types.h"
#include "load_TrtEngine_runtime.h"

enum class LaneState {
    Idle,
    Running,
    Ready,
    Failed
};

struct FrameResult {
    std::uint64_t frame_id{};
    std::vector<Detection> detections;
};

struct LaneConfig {
    float confidence_threshold{0.25f};
    int num_classes{80};
};

// One in-flight frame per lane. The shared model must outlive this object.
// Call these methods from one host thread, or serialize access externally.
class lane {
public:
    explicit lane(const SharedTrtEngine& model, LaneConfig config = {});
    ~lane();
    lane(const lane&) = delete;
    lane& operator=(const lane&) = delete;

    // Copies CV_8UC3 BGR pixels into lane-owned memory before returning.
    // False means the lane is not idle. CUDA/TRT errors throw and fail the lane.
    bool submit(const cv::Mat& image, std::uint64_t frame_id);
    LaneState query();
    // Nonblocking: false leaves result unchanged; true returns owned CPU data.
    bool collect(FrameResult& result);

private:
    void release() noexcept;

    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_{};
    BufferManager buffers_;
    CudaPreprocessor preprocessor_;

    void* host_raw_{nullptr};       // Pinned CPU input staging.
    std::size_t raw_capacity_{0};
    void* device_packet_{nullptr};  // int count followed by Detection[N].
    void* host_packet_{nullptr};    // Pinned CPU download destination.
    std::size_t packet_bytes_{0};

    int input_width_{0};
    int input_height_{0};
    int num_boxes_{0};
    std::uint64_t frame_id_{0};
    LaneConfig config_;

    cudaEvent_t start_event_{};
    cudaEvent_t h2d_event_{};
    cudaEvent_t execute_event_{};
    cudaEvent_t d2h_event_{};
    cudaEvent_t done_event_{};
    LaneState state_{LaneState::Idle};
};
