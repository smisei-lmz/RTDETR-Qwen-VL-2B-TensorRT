#include "laneManager.h"
#include "CudaPostprocess.h"

#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {
void checkCudaEvent(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

// Match the existing COCO RT-DETR benchmark.
constexpr float confidence_threshold = 0.25f;
constexpr int num_classes = 80;
static_assert(alignof(Detection) <= alignof(int), "Detection packet alignment");
}

lane::lane(const SharedTrtEngine& model) {
    if (!model.engine()) {
        throw std::runtime_error("Shared engine is not loaded");
    }
    input_width_ = model.inputWidth();
    input_height_ = model.inputHeight();
    num_boxes_ = model.outputSize() / 6;
    if (num_boxes_ <= 0 || num_boxes_ > 1024) {
        throw std::invalid_argument("CUDA postprocessing requires 1 <= N <= 1024");
    }
    packet_bytes_ = sizeof(int) + static_cast<std::size_t>(num_boxes_) * sizeof(Detection);

    try {
        context_.reset(model.engine()->createExecutionContext());
        if (!context_) {
            throw std::runtime_error("Cannot create execution context");
        }
        checkCudaEvent(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        checkCudaEvent(cudaEventCreateWithFlags(&done_event_, cudaEventDisableTiming));
        checkCudaEvent(cudaEventCreate(&start_event_));
        checkCudaEvent(cudaEventCreate(&h2d_event_));
        checkCudaEvent(cudaEventCreate(&execute_event_));
        checkCudaEvent(cudaEventCreate(&d2h_event_));

        buffers_.allocate(
            static_cast<std::size_t>(model.inputSize()) * sizeof(float),
            static_cast<std::size_t>(model.outputSize()) * sizeof(float));
        if (!context_->setTensorAddress(model.inputName().c_str(), buffers_.input_device()) ||
            !context_->setTensorAddress(model.outputName().c_str(), buffers_.output_device())) {
            throw std::runtime_error("Failed to bind tensor addresses");
        }

        checkCudaEvent(cudaMalloc(&device_packet_, packet_bytes_));
        checkCudaEvent(cudaMallocHost(&host_packet_, packet_bytes_));
        // The decoder writes count and valid detections. Initialize unused slots once.
        checkCudaEvent(cudaMemsetAsync(device_packet_, 0, packet_bytes_, stream_));
    } catch (...) {
        release();
        throw;
    }
}

lane::~lane() {
    release();
}

bool lane::submit(const cv::Mat& image, std::uint64_t frame_id) {
    if (state_ != LaneState::Idle) {
        return false;
    }
    if (image.empty() || image.dims != 2 || image.type() != CV_8UC3) {
        throw std::invalid_argument("Expected a nonempty CV_8UC3 BGR image");
    }
    const auto row_bytes = static_cast<std::size_t>(image.cols) * 3;
    // The current resize kernel uses int offsets into the packed source image.
    if (row_bytes > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        static_cast<std::size_t>(image.rows) >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) / row_bytes) {
        throw std::invalid_argument("Image is too large for the CUDA resize kernel");
    }
    const auto raw_bytes = row_bytes * static_cast<std::size_t>(image.rows);

    try {
        // Only an idle lane may resize or overwrite its buffers.
        if (raw_bytes > raw_capacity_) {
            if (host_raw_) {
                checkCudaEvent(cudaFreeHost(host_raw_));
                host_raw_ = nullptr;
                raw_capacity_ = 0;
            }
            checkCudaEvent(cudaMallocHost(&host_raw_, raw_bytes));
            raw_capacity_ = raw_bytes;
        }
        if (!buffers_.ensureRawCapacity(raw_bytes)) {
            throw std::runtime_error("Cannot allocate GPU image buffer");
        }
        checkCudaEvent(preprocessor_.configureResize(
            image.cols, image.rows, input_width_, input_height_));

        // Own the pixels before returning, including non-contiguous cv::Mat ROIs.
        if (image.isContinuous()) {
            std::memcpy(host_raw_, image.data, raw_bytes);
        } else {
            auto* destination = static_cast<unsigned char*>(host_raw_);
            for (int y = 0; y < image.rows; ++y) {
                std::memcpy(destination + static_cast<std::size_t>(y) * row_bytes,
                            image.ptr<unsigned char>(y), row_bytes);
            }
        }
        frame_id_ = frame_id;
        // Set before the first enqueue: a partial submission must never be reused.
        state_ = LaneState::Running;

        checkCudaEvent(cudaEventRecord(start_event_, stream_));
        checkCudaEvent(cudaMemcpyAsync(buffers_.raw_device(), host_raw_, raw_bytes,
                                       cudaMemcpyHostToDevice, stream_));
        checkCudaEvent(cudaEventRecord(h2d_event_, stream_));

        checkCudaEvent(preprocessor_.resizeNormalize(
            static_cast<const unsigned char*>(buffers_.raw_device()),
            static_cast<int>(row_bytes),
            static_cast<float*>(buffers_.input_device()), stream_));
        if (!context_->enqueueV3(stream_)) {
            throw std::runtime_error("TensorRT enqueue failed");
        }
        checkCudaEvent(cudaEventRecord(execute_event_, stream_));

        auto* device_detections = reinterpret_cast<Detection*>(
            static_cast<char*>(device_packet_) + sizeof(int));
        checkCudaEvent(CudaPostprocessor::decode(
            static_cast<const float*>(buffers_.output_device()), num_boxes_,
            confidence_threshold, image.cols, image.rows, num_classes,
            device_detections, static_cast<int*>(device_packet_), stream_));

        checkCudaEvent(cudaMemcpyAsync(host_packet_, device_packet_, packet_bytes_,
                                       cudaMemcpyDeviceToHost, stream_));
        checkCudaEvent(cudaEventRecord(d2h_event_, stream_));
        checkCudaEvent(cudaEventRecord(done_event_, stream_));
        return true;
    } catch (...) {
        // Earlier copies/kernels may still be running; destruction waits for them.
        state_ = LaneState::Failed;
        throw;
    }
}

LaneState lane::query() {
    if (state_ != LaneState::Running) {
        return state_;
    }
    const auto status = cudaEventQuery(done_event_);
    if (status == cudaSuccess) {
        state_ = LaneState::Ready;
    } else if (status != cudaErrorNotReady) {
        state_ = LaneState::Failed;
        checkCudaEvent(status);
    }
    return state_;
}

bool lane::collect(FrameResult& result) {
    // Querying is nonblocking and also permits collect() without a separate query().
    if (query() != LaneState::Ready) {
        return false;
    }
    const int count = *static_cast<const int*>(host_packet_);
    if (count < 0 || count > num_boxes_) {
        state_ = LaneState::Failed;
        throw std::runtime_error("Invalid GPU detection count");
    }

    const auto* detections = reinterpret_cast<const Detection*>(
        static_cast<const char*>(host_packet_) + sizeof(int));
    FrameResult completed;
    completed.frame_id = frame_id_;
    completed.detections.assign(detections, detections + count);

    // Do not recycle until the caller has a result independent of lane buffers.
    result = std::move(completed);
    state_ = LaneState::Idle;
    return true;
}

void lane::release() noexcept {
    if (stream_) {
        cudaStreamSynchronize(stream_);
    }
    context_.reset();
    buffers_.release();
    if (host_raw_) {
        cudaFreeHost(host_raw_);
        host_raw_ = nullptr;
        raw_capacity_ = 0;
    }
    if (device_packet_) {
        cudaFree(device_packet_);
        device_packet_ = nullptr;
    }
    if (host_packet_) {
        cudaFreeHost(host_packet_);
        host_packet_ = nullptr;
    }
    for (auto* event : {&start_event_, &h2d_event_, &execute_event_,
                        &d2h_event_, &done_event_}) {
        if (*event) {
            cudaEventDestroy(*event);
            *event = nullptr;
        }
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}
