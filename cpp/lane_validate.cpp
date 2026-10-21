#include "laneManager.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}
std::size_t rssBytes() {
    std::ifstream input("/proc/self/status");
    std::string key, line;
    while (input >> key) {
        if (key == "VmRSS:") {
            std::size_t kb;
            require(bool(input >> kb), "Cannot read VmRSS");
            return kb * 1024;
        }
        std::getline(input, line);
    }
    throw std::runtime_error("VmRSS not available");
}
struct Memory {
    std::size_t rss, gpu_free, gpu_total;
};
Memory memory() {
    Memory value{rssBytes(), 0, 0};
    check(cudaMemGetInfo(&value.gpu_free, &value.gpu_total));
    return value;
}
FrameResult infer(lane& current, const cv::Mat& image, std::uint64_t id) {
    require(current.submit(image, id), "Idle lane rejected input");
    const auto deadline = Clock::now() + std::chrono::seconds(60);
    FrameResult result;
    while (!current.collect(result)) {
        require(Clock::now() < deadline, "GPU completion timed out");
        // Only CUDA completion events authorize reading results or recycling.
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    require(result.frame_id == id && current.query() == LaneState::Idle,
            "Frame identity or lane lifecycle failure");
    for (const auto& det : result.detections) {
        require(det.class_id >= 0 && det.class_id < 80 &&
                std::isfinite(det.confidence) && std::isfinite(det.x1) &&
                std::isfinite(det.y1) && std::isfinite(det.x2) && std::isfinite(det.y2) &&
                det.x2 > det.x1 && det.y2 > det.y1,
                "Invalid detection");
    }
    return result;
}
void compare(const FrameResult& actual, const FrameResult& expected) {
    require(actual.detections.size() == expected.detections.size(),
            "Repeated input changed detection count");
    for (std::size_t i = 0; i < actual.detections.size(); ++i) {
        const auto& a = actual.detections[i];
        const auto& b = expected.detections[i];
        require(a.class_id == b.class_id && std::abs(a.confidence - b.confidence) <= 1e-5f &&
                std::abs(a.x1 - b.x1) <= 1e-3f && std::abs(a.y1 - b.y1) <= 1e-3f &&
                std::abs(a.x2 - b.x2) <= 1e-3f && std::abs(a.y2 - b.y2) <= 1e-3f,
                "Repeated input changed detection values");
    }
}
std::ofstream output(const std::filesystem::path& path) {
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream result(path);
    require(bool(result), "Cannot open output file");
    result << std::setprecision(12);
    return result;
}
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: trt_lane_validate --engine PATH --images LIST --prefix PATH "
                         "[--threshold 0.001] [--duration-seconds 0] [--warmup 30]\n"
                         "duration=0: dataset predictions; duration>0: repeated-input stability.\n";
            return 0;
        }
        std::filesystem::path engine_path, image_list, prefix;
        float threshold = 0.001f;
        int duration = 0, warmup = 30;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            require(i + 1 < argc, "Missing option value");
            const std::string value = argv[++i];
            if (key == "--engine") engine_path = value;
            else if (key == "--images") image_list = value;
            else if (key == "--prefix") prefix = value;
            else if (key == "--threshold") threshold = std::stof(value);
            else if (key == "--duration-seconds") duration = std::stoi(value);
            else if (key == "--warmup") warmup = std::stoi(value);
            else throw std::invalid_argument("Unknown option: " + key);
        }
        require(!engine_path.empty() && !image_list.empty() && !prefix.empty() &&
                duration >= 0 && warmup >= 0, "Invalid options");
        std::ifstream list(image_list);
        require(bool(list), "Cannot read image list");
        std::vector<std::string> paths;
        std::string path;
        while (std::getline(list, path)) if (!path.empty()) paths.push_back(path);
        require(!paths.empty(), "Image list is empty");
        cv::setNumThreads(1);
        // Stability preloads its fixed images; dataset mode decodes one at a time.
        std::vector<cv::Mat> images;
        if (duration > 0) {
            for (const auto& name : paths) {
                images.push_back(cv::imread(name));
                require(!images.back().empty(), "Cannot read stability image");
            }
        }
        SharedTrtEngine model;
        require(model.loadEngine(engine_path.string()), "Cannot load model");
        const auto engine_memory = memory();
        auto current = std::make_unique<lane>(model, LaneConfig{threshold, 80});
        auto first = images.empty() ? cv::imread(paths.front()) : images.front();
        require(!first.empty(), "Cannot read first image");
        for (int i = 0; i < warmup; ++i) infer(*current, first, 0);
        std::vector<FrameResult> expected;
        for (const auto& image : images) expected.push_back(infer(*current, image, 0));

        auto samples = output(prefix.string() + ".samples.csv");
        samples << "frame_id,image_index,latency_ms,detections\n";
        std::ofstream predictions;
        if (duration == 0) {
            predictions = output(prefix.string() + ".predictions.csv");
            predictions << "image_index,class_id,score,x1,y1,x2,y2\n";
        }
        auto telemetry = output(prefix.string() + ".memory.csv");
        telemetry << "elapsed_s,completed,rss_bytes,device_free_bytes,device_total_bytes\n";
        const auto before = memory();
        const auto begin = Clock::now();
        auto last_memory = begin;
        std::uint64_t completed = 0;
        auto record_memory = [&] {
            const auto value = memory();
            telemetry << milliseconds(begin, Clock::now()) / 1000 << ',' << completed << ','
                      << value.rss << ',' << value.gpu_free << ',' << value.gpu_total << '\n';
            telemetry.flush();
        };
        record_memory();
        while ((duration == 0 && completed < paths.size()) ||
               (duration > 0 && Clock::now() - begin < std::chrono::seconds(duration))) {
            const auto index = static_cast<std::size_t>(completed % paths.size());
            const auto image = duration > 0 ? images[index] : cv::imread(paths[index]);
            require(!image.empty(), "Cannot decode dataset image");
            const auto submitted = Clock::now();
            const auto result = infer(*current, image, completed);
            const auto latency = milliseconds(submitted, Clock::now());
            if (duration > 0) compare(result, expected[index]);
            for (const auto& det : result.detections) {
                if (duration == 0) predictions << index << ',' << det.class_id << ','
                    << det.confidence << ',' << det.x1 << ',' << det.y1 << ','
                    << det.x2 << ',' << det.y2 << '\n';
            }
            samples << completed << ',' << index << ',' << latency << ','
                    << result.detections.size() << '\n';
            ++completed;
            if (Clock::now() - last_memory >= std::chrono::seconds(10)) {
                record_memory();
                samples.flush();
                last_memory = Clock::now();
                std::cout << "elapsed_s=" << milliseconds(begin, last_memory) / 1000
                          << " completed=" << completed << std::endl;
            }
        }
        const auto elapsed_ms = milliseconds(begin, Clock::now());
        record_memory();
        const auto after = memory();
        require(current->query() == LaneState::Idle, "Lane has pending work at teardown");
        current.reset();  // Wait and free the lane before measuring released resources.
        const auto released = memory();
        samples.flush();
        telemetry.flush();
        if (predictions.is_open()) { predictions.flush(); require(bool(predictions), "Prediction write failed"); }
        require(bool(samples) && bool(telemetry), "Log write failed");
        auto summary = output(prefix.string() + ".summary.json");
        summary << "{\"passed\":true,\"mode\":"
                << std::quoted(duration ? "stability" : "dataset")
                << ",\"requested_duration_s\":" << duration << ",\"elapsed_s\":" << elapsed_ms / 1000
                << ",\"submitted\":" << completed << ",\"completed\":" << completed
                << ",\"failed\":0,\"dropped\":0,\"lanes\":1,\"warmup\":" << warmup
                << ",\"threshold\":" << threshold << ",\"image_count\":" << paths.size()
                << ",\"all_frames_collected\":true,\"lane_destroyed\":true"
                << ",\"rss_before_bytes\":" << before.rss << ",\"rss_after_bytes\":" << after.rss
                << ",\"device_free_before_bytes\":" << before.gpu_free
                << ",\"device_free_after_bytes\":" << after.gpu_free
                << ",\"device_free_engine_only_bytes\":" << engine_memory.gpu_free
                << ",\"device_free_after_lane_teardown_bytes\":" << released.gpu_free
                << ",\"device_memory_scope\":\"whole device; may include other processes\""
                << ",\"latency_scope\":\"submit through collect; excludes image decode and CSV writes\"}\n";
        summary.flush();
        require(bool(summary), "Summary write failed");
        std::cout << "PASS: completed=" << completed << " elapsed_s=" << elapsed_ms / 1000
                  << " all frames collected; lane destroyed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
