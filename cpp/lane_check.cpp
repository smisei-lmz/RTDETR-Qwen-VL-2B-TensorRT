#include "laneManager.h"
#include "Postprocess.h"
#include "Preprocess.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Stream {
    cudaStream_t value{};
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() {
        cudaStreamSynchronize(value);
        cudaStreamDestroy(value);
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

// Independent CPU preprocessing/postprocessing reference, sharing only the engine.
std::vector<Detection> reference(const SharedTrtEngine& model, const cv::Mat& image,
                                 float threshold = 0.25f) {
    std::vector<float> input(model.inputSize()), output(model.outputSize());
    OpenCVPreprocessWorkspace workspace;
    workspace.create(model.inputWidth(), model.inputHeight());
    Preprocessor::preprocessOpenCV(
        image, input.data(), model.inputWidth(), model.inputHeight(), workspace);

    BufferManager buffers;
    buffers.allocate(input.size() * sizeof(float), output.size() * sizeof(float));
    std::unique_ptr<nvinfer1::IExecutionContext> context{
        model.engine()->createExecutionContext()
    };
    require(bool(context), "Cannot create reference context");
    Stream stream;  // Waits before context, buffers, or host arrays are destroyed.
    require(context->setTensorAddress(model.inputName().c_str(), buffers.input_device()) &&
            context->setTensorAddress(model.outputName().c_str(), buffers.output_device()),
            "Cannot bind reference tensors");
    check(cudaMemcpyAsync(buffers.input_device(), input.data(), input.size() * sizeof(float),
                          cudaMemcpyHostToDevice, stream.value));
    require(context->enqueueV3(stream.value), "Reference inference failed");
    check(cudaMemcpyAsync(output.data(), buffers.output_device(), output.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, stream.value));
    check(cudaStreamSynchronize(stream.value));
    return Postprocessor::decode(output.data(), model.outputSize() / 6, threshold,
                                 image.cols, image.rows, 80);
}

void compare(const FrameResult& result, std::uint64_t id,
             const std::vector<Detection>& expected) {
    require(result.frame_id == id, "Frame ID mismatch");
    require(result.detections.size() == expected.size(), "Detection count mismatch");
    float max_coordinate_difference = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto& actual = result.detections[i];
        const auto& wanted = expected[i];
        require(actual.class_id == wanted.class_id, "Class ID mismatch");
        require(std::isfinite(actual.confidence) &&
                std::abs(actual.confidence - wanted.confidence) <= 1e-5f,
                "Confidence mismatch");
        const float actual_coordinates[] = {actual.x1, actual.y1, actual.x2, actual.y2};
        const float expected_coordinates[] = {wanted.x1, wanted.y1, wanted.x2, wanted.y2};
        for (int j = 0; j < 4; ++j) {
            const float difference = std::abs(actual_coordinates[j] - expected_coordinates[j]);
            require(std::isfinite(actual_coordinates[j]) && difference <= 1e-3f,
                    "Box coordinate mismatch");
            max_coordinate_difference = std::max(max_coordinate_difference, difference);
        }
    }
    std::cout << "Frame " << id << ": " << result.detections.size()
              << " detections; max coordinate difference="
              << max_coordinate_difference << '\n';
}

void waitReady(lane& value) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (true) {
        const auto state = value.query();
        if (state == LaneState::Ready) return;
        require(state == LaneState::Running, "Unexpected lane state");
        // Deadline is a test failure only; it never authorizes buffer reuse.
        require(std::chrono::steady_clock::now() < deadline, "Completion event timed out");
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

FrameResult waitCollect(lane& value) {
    FrameResult result;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (!value.collect(result)) {
        require(std::chrono::steady_clock::now() < deadline, "Collection timed out");
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return result;
}
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: trt_lane_check [engine image_a image_b]\n";
            return 0;
        }
        require(argc == 1 || argc == 4, "Expected engine and two images, or no arguments");
        const std::filesystem::path root(RTDETR_PROJECT_DIR);
        const auto engine_path = argc == 4 ? std::filesystem::path(argv[1])
            : root / "models/rtdetr-l-fp32mix16-trt10.engine";
        const auto image_a_path = argc == 4 ? std::filesystem::path(argv[2])
            : root / "ultralytics/coco/images/val2017/000000147740.jpg";
        const auto image_b_path = argc == 4 ? std::filesystem::path(argv[3])
            : root / "ultralytics/coco/images/val2017/000000057150.jpg";
        cv::setNumThreads(1);
        const auto image_a = cv::imread(image_a_path.string());
        const auto image_b = cv::imread(image_b_path.string());
        require(!image_a.empty() && !image_b.empty(), "Cannot read test images");

        SharedTrtEngine model;
        require(model.loadEngine(engine_path.string()), "Cannot load model");
        bool invalid_config_rejected = false;
        try {
            lane invalid(model, LaneConfig{-0.1f, 80});
        } catch (const std::invalid_argument&) {
            invalid_config_rejected = true;
        }
        require(invalid_config_rejected, "Invalid threshold was accepted");
        {
            lane evaluation(model, LaneConfig{0.001f, 80});
            require(evaluation.submit(image_a, 10), "Cannot submit evaluation frame");
            compare(waitCollect(evaluation), 10, reference(model, image_a, 0.001f));
        }
        const auto expected_a = reference(model, image_a);
        const auto expected_b = reference(model, image_b);
        lane first(model), second(model);
        require(first.query() == LaneState::Idle, "New lane is not idle");
        FrameResult unchanged;
        unchanged.frame_id = 999;
        require(!first.collect(unchanged) && unchanged.frame_id == 999,
                "Idle collection modified output");
        for (const auto& invalid : {cv::Mat{}, cv::Mat(2, 2, CV_8UC1)}) {
            bool rejected = false;
            try {
                first.submit(invalid, 0);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            require(rejected && first.query() == LaneState::Idle,
                    "Invalid input was accepted or poisoned the idle lane");
        }

        auto source_a = image_a.clone();
        auto source_b = image_b.clone();
        require(first.submit(source_a, 11), "Cannot submit first frame");
        require(second.submit(source_b, 12), "Cannot submit second frame");
        require(!first.submit(image_b, 99), "Busy lane accepted another frame");
        // The caller may immediately reuse its input storage after submit().
        source_a.setTo(cv::Scalar::all(0));
        source_b.setTo(cv::Scalar::all(0));
        waitReady(first);
        waitReady(second);
        require(!first.submit(image_b, 99), "Ready lane overwrote an uncollected result");
        FrameResult result_a, result_b;
        require(first.collect(result_a) && second.collect(result_b), "Cannot collect ready frames");
        compare(result_a, 11, expected_a);
        compare(result_b, 12, expected_b);
        require(first.query() == LaneState::Idle && second.query() == LaneState::Idle,
                "Collected lane was not recycled");
        require(!first.collect(result_a) && result_a.frame_id == 11,
                "Double collection succeeded or modified output");

        cv::Mat padded(image_a.rows + 4, image_a.cols + 8, CV_8UC3);
        auto roi = padded(cv::Rect(3, 2, image_a.cols, image_a.rows));
        image_a.copyTo(roi);
        require(!roi.isContinuous(), "Test ROI unexpectedly contiguous");
        require(first.submit(roi, 13), "Cannot submit non-contiguous ROI");
        // Reconfigure a reused lane when the source resolution changes.
        require(second.submit(image_a, 14), "Cannot reuse second lane");
        padded.setTo(cv::Scalar::all(0));
        compare(waitCollect(first), 13, expected_a);
        compare(waitCollect(second), 14, expected_a);
        compare(result_a, 11, expected_a); // Prior result still owns its data.

        {
            lane pending(model);
            require(pending.submit(image_b, 15), "Cannot submit teardown frame");
            // No query/collect: the destructor must wait before releasing resources.
        }
        check(cudaDeviceSynchronize());
        std::cout << "PASS: threshold configuration, two lanes, CPU parity, input ownership, ROI strides, state gating, "
                     "reuse, independent results, and pending-work teardown\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
