#include "laneManager.h"
#include "nvtx_range.h"

#include <cuda_profiler_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
struct Sample {
    int round, lanes;
    std::uint64_t frame_id;
    double submitted_ms, collected_ms, latency_ms;
};
struct Run {
    int round, lanes, completed;
    double wall_ms, fps, cpu_ms, mean_ms, p50_ms, p95_ms, p99_ms;
};
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
double ms(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}
double percentile(const std::vector<double>& sorted, double p) {
    return sorted[static_cast<std::size_t>(std::ceil(p * sorted.size())) - 1];
}

// Closed-loop load: submit only to idle lanes; no upstream arrival queue or drops.
Run execute(std::vector<std::unique_ptr<lane>>& lanes, const cv::Mat& image,
            int frames, int poll_us, int round, std::vector<Sample>* samples,
            int per_lane_limit = 0) {
    std::vector<Clock::time_point> submitted(frames);
    std::vector<bool> seen(frames, false);
    std::vector<int> submitted_per_lane(lanes.size(), 0);
    std::vector<double> latencies;
    latencies.reserve(frames);
    int next = 0, completed = 0;
    const auto cpu_begin = std::clock();
    const auto begin = Clock::now();
    auto finished = begin;
    auto last_progress = begin;

    while (completed < frames) {
        bool progress = false;
        for (std::size_t lane_id = 0; lane_id < lanes.size(); ++lane_id) {
            auto& current = lanes[lane_id];
            FrameResult result;
            if (current->collect(result)) {
                const auto collected = Clock::now();
                require(result.frame_id < static_cast<std::uint64_t>(next),
                        "Unexpected frame identity");
                const auto id = static_cast<std::size_t>(result.frame_id);
                require(!seen[id], "Duplicate frame result");
                seen[id] = true;
                const auto latency = ms(submitted[id], collected);
                latencies.push_back(latency);
                if (samples) {
                    samples->push_back({round, static_cast<int>(lanes.size()), result.frame_id,
                                       ms(begin, submitted[id]), ms(begin, collected), latency});
                }
                ++completed;
                finished = collected;
                progress = true;
            }
            const auto state = current->query();
            require(state != LaneState::Failed, "Lane failed during measurement");
            if (state == LaneState::Idle && next < frames &&
                (!per_lane_limit || submitted_per_lane[lane_id] < per_lane_limit)) {
                submitted[next] = Clock::now();
                require(current->submit(image, static_cast<std::uint64_t>(next)),
                        "Idle lane rejected a frame");
                ++next;
                ++submitted_per_lane[lane_id];
                progress = true;
            }
        }
        if (progress) {
            last_progress = Clock::now();
        } else {
            // A timeout aborts the run; it never authorizes buffer reuse.
            require(Clock::now() - last_progress < std::chrono::seconds(60),
                    "No completed frames for 60 seconds");
            if (poll_us > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(poll_us));
            } else {
                std::this_thread::yield();
            }
        }
    }

    const auto cpu_end = std::clock();
    const auto wall_ms = ms(begin, finished);
    const auto mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) / frames;
    std::sort(latencies.begin(), latencies.end());
    return {round, static_cast<int>(lanes.size()), completed, wall_ms,
            frames * 1000.0 / wall_ms,
            1000.0 * (cpu_end - cpu_begin) / CLOCKS_PER_SEC,
            mean, percentile(latencies, 0.50), percentile(latencies, 0.95),
            percentile(latencies, 0.99)};
}

void writeRun(std::ostream& output, const Run& run) {
    output << "{\"round\":" << run.round << ",\"lanes\":" << run.lanes
           << ",\"submitted\":" << run.completed << ",\"completed\":" << run.completed
           << ",\"dropped\":0,\"failed\":0,\"wall_ms\":" << run.wall_ms
           << ",\"fps\":" << run.fps << ",\"process_cpu_ms\":" << run.cpu_ms
           << ",\"latency_mean_ms\":" << run.mean_ms << ",\"latency_p50_ms\":" << run.p50_ms
           << ",\"latency_p95_ms\":" << run.p95_ms << ",\"latency_p99_ms\":" << run.p99_ms
           << "}";
}
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: trt_lane_benchmark [engine image output_prefix] "
                         "[--frames N] [--warmup N-per-lane] [--rounds N] "
                         "[--lanes both|1|2] [--poll-us N] [--profile 0|1]\n";
            return 0;
        }
        const std::filesystem::path root(RTDETR_PROJECT_DIR);
        auto engine_path = root / "models/rtdetr-l-fp32mix16-trt10.engine";
        auto image_path = root / "ultralytics/coco/images/val2017/000000147740.jpg";
        auto prefix = root / "results/lane_benchmark_fp16";
        int frames = 500, warmup = 30, rounds = 5, poll_us = 50, profile = 0, arg = 1;
        std::string mode = "both";
        if (arg < argc && std::string(argv[arg]).rfind("--", 0) != 0) {
            require(argc >= 4, "Expected engine, image and output prefix");
            engine_path = argv[arg++];
            image_path = argv[arg++];
            prefix = argv[arg++];
        }
        for (; arg < argc; ++arg) {
            const std::string key = argv[arg];
            require(arg + 1 < argc, "Missing option value");
            const std::string value = argv[++arg];
            if (key == "--lanes") mode = value;
            else if (key == "--frames") frames = std::stoi(value);
            else if (key == "--warmup") warmup = std::stoi(value);
            else if (key == "--rounds") rounds = std::stoi(value);
            else if (key == "--poll-us") poll_us = std::stoi(value);
            else if (key == "--profile") profile = std::stoi(value);
            else throw std::invalid_argument("Unknown option: " + key);
        }
        require(frames > 0 && warmup >= 0 && warmup <= 1000000 &&
                rounds > 0 && poll_us >= 0 && (profile == 0 || profile == 1),
                "Invalid measurement counts");
        require(mode == "both" || mode == "1" || mode == "2", "Invalid lane count");
        require(!profile || (mode != "both" && rounds == 1),
                "Profiling requires one lane configuration and one round");
        cv::setNumThreads(1);
        const auto image = cv::imread(image_path.string());
        require(!image.empty(), "Cannot read input image");
        SharedTrtEngine model;
        require(model.loadEngine(engine_path.string()), "Cannot load engine");

        std::vector<Run> runs;
        std::vector<Sample> samples;
        samples.reserve(static_cast<std::size_t>(frames) * rounds * (mode == "both" ? 2 : 1));
        for (int round = 0; round < rounds; ++round) {
            std::vector<int> order = mode == "both"
                ? (round % 2 == 0 ? std::vector<int>{1, 2} : std::vector<int>{2, 1})
                : std::vector<int>{std::stoi(mode)};
            for (int count : order) {
                std::vector<std::unique_ptr<lane>> lanes;
                for (int i = 0; i < count; ++i) lanes.push_back(std::make_unique<lane>(model));
                if (warmup > 0) {
                    NvtxRange range("warmup");
                    execute(lanes, image, warmup * count, poll_us, round, nullptr, warmup);
                }
                if (profile) check(cudaProfilerStart());
                Run run{};
                {
                    NvtxRange range(count == 1 ? "steady_lane1" : "steady_lane2");
                    run = execute(lanes, image, frames, poll_us, round, &samples);
                }
                if (profile) check(cudaProfilerStop());
                runs.push_back(run);
                std::cout << "Round " << round + 1 << ", lanes=" << count
                          << ": FPS=" << run.fps << ", latency mean/P95/P99="
                          << run.mean_ms << "/" << run.p95_ms << "/" << run.p99_ms
                          << " ms, completed=" << run.completed << '\n';
            }
        }

        if (!prefix.parent_path().empty()) std::filesystem::create_directories(prefix.parent_path());
        std::ofstream csv(prefix.string() + ".samples.csv");
        std::ofstream summary(prefix.string() + ".summary.json");
        require(bool(csv) && bool(summary), "Cannot open output files");
        csv << std::setprecision(12)
            << "round,lanes,frame_id,submitted_ms,collected_ms,latency_ms\n";
        for (const auto& sample : samples) {
            csv << sample.round << ',' << sample.lanes << ',' << sample.frame_id << ','
                << sample.submitted_ms << ',' << sample.collected_ms << ','
                << sample.latency_ms << '\n';
        }
        summary << std::setprecision(12)
                << "{\"engine\":" << std::quoted(engine_path.string())
                << ",\"image\":" << std::quoted(image_path.string())
                << ",\"source_width\":" << image.cols << ",\"source_height\":" << image.rows
                << ",\"frames_per_round\":" << frames << ",\"warmup_per_lane\":" << warmup
                << ",\"rounds\":" << rounds << ",\"poll_us\":" << poll_us
                << ",\"profiled\":" << (profile ? "true" : "false")
                << ",\"scope\":\"closed-loop fixed-image throughput; latency from submit start "
                   "to collected CPU result; excludes decode, admission wait and ROS\""
                << ",\"percentile_method\":\"nearest_rank\",\"runs\":[";
        for (std::size_t i = 0; i < runs.size(); ++i) {
            if (i) summary << ',';
            writeRun(summary, runs[i]);
        }
        summary << "],\"aggregate\":[";
        bool first = true;
        for (int count : {1, 2}) {
            int completed = 0;
            double wall = 0, cpu = 0;
            std::vector<double> latencies;
            for (const auto& run : runs) {
                if (run.lanes != count) continue;
                completed += run.completed;
                wall += run.wall_ms;
                cpu += run.cpu_ms;
            }
            if (!completed) continue;
            for (const auto& sample : samples) {
                if (sample.lanes == count) latencies.push_back(sample.latency_ms);
            }
            const auto mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) /
                              latencies.size();
            std::sort(latencies.begin(), latencies.end());
            if (!first) summary << ',';
            first = false;
            writeRun(summary, {-1, count, completed, wall, completed * 1000.0 / wall,
                               cpu, mean, percentile(latencies, 0.50),
                               percentile(latencies, 0.95), percentile(latencies, 0.99)});
        }
        summary << "]}\n";
        require(bool(csv) && bool(summary), "Cannot write measurement results");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
