#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <opencv2/opencv.hpp>
#include "TrtEngine.h"
#include "Preprocess.h"
#include "Postprocess.h"
#include "CudaPreprocess.h"
#include "CudaPostprocess.h"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
struct Memory {
    void* ptr{nullptr}; bool host;
    Memory(size_t bytes, bool pinned) : host(pinned) {
        check(host ? cudaMallocHost(&ptr, bytes) : cudaMalloc(&ptr, bytes));
    }
    ~Memory() { if (ptr) { if (host) cudaFreeHost(ptr); else cudaFree(ptr); } }
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;
};
struct Events {
    std::array<cudaEvent_t, 6> e{};
    Events() {
        try { for (auto& v : e) check(cudaEventCreate(&v)); }
        catch (...) { for (auto v : e) if (v) cudaEventDestroy(v); throw; }
    }
    ~Events() { for (auto v : e) cudaEventDestroy(v); }
    void record(int i, cudaStream_t stream) { check(cudaEventRecord(e[i], stream)); }
    double elapsed(int a, int b) {
        float value; check(cudaEventElapsedTime(&value, e[a], e[b])); return value;
    }
};
struct StreamCompletionGuard {
    cudaStream_t stream;
    // Also protect buffers during stack unwinding if a later enqueue fails.
    ~StreamCompletionGuard() { cudaStreamSynchronize(stream); }
};
struct Sample {
    double cpu_prepare{}, h2d{}, gpu_pre{}, infer{}, gpu_post{}, d2h{}, cpu_post{}, total{};
};
struct Stats { double mean, median, p95; };
Stats stats(const std::vector<Sample>& samples, double Sample::*field) {
    std::vector<double> values;
    for (const auto& s : samples) values.push_back(s.*field);
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    const double median = n % 2 ? values[n/2] : (values[n/2-1] + values[n/2]) / 2;
    return {std::accumulate(values.begin(), values.end(), 0.0) / n, median,
            values[static_cast<size_t>(std::ceil(n * 0.95)) - 1]};
}
const std::array<std::pair<const char*, double Sample::*>, 8> fields{{
    {"cpu_prepare", &Sample::cpu_prepare}, {"h2d", &Sample::h2d},
    {"gpu_pre", &Sample::gpu_pre}, {"infer", &Sample::infer},
    {"gpu_post", &Sample::gpu_post}, {"d2h", &Sample::d2h},
    {"cpu_post", &Sample::cpu_post}, {"total", &Sample::total}}};
void writeStats(std::ostream& file, const std::vector<Sample>& samples) {
    file << "{";
    for (size_t i = 0; i < fields.size(); ++i) {
        const auto s = stats(samples, fields[i].second);
        file << (i ? "," : "") << '\n' << std::quoted(fields[i].first)
             << ": {\"mean_ms\":" << s.mean << ",\"median_ms\":" << s.median
             << ",\"p95_ms\":" << s.p95 << "}";
    }
    file << "\n}";
}
float difference(const std::vector<float>& a, const std::vector<float>& b) {
    float result = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) throw std::runtime_error("Non-finite tensor");
        result = std::max(result, std::abs(a[i] - b[i]));
    }
    return result;
}
}

int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: trt_benchmark [engine image output_prefix] [--iterations N] [--warmup N] [--opencv-threads N] [--cpu-staging 0|1]\n";
            return 0;
        }
        const fs::path root(RTDETR_PROJECT_DIR);
        fs::path engine_path = root / "models/rtdetr-l-int8-box-add6-fp32.engine";
        fs::path image_path = root / "ultralytics/coco/images/val2017/000000147740.jpg";
        fs::path prefix = root / "results/benchmark_single_lane";
        int iterations = 200, warmup = 30, threads = 1, cpu_staging = 0, arg = 1;
        if (arg < argc && std::string(argv[arg]).rfind("--", 0) != 0) {
            if (argc < 4) throw std::runtime_error("Expected engine, image and output prefix");
            engine_path = argv[arg++]; image_path = argv[arg++]; prefix = argv[arg++];
        }
        for (; arg < argc; ++arg) {
            const std::string key(argv[arg]);
            if (arg + 1 >= argc) throw std::runtime_error("Incomplete option: " + key);
            const int value = std::stoi(argv[++arg]);
            if (key == "--iterations") iterations = value;
            else if (key == "--warmup") warmup = value;
            else if (key == "--opencv-threads") threads = value;
            else if (key == "--cpu-staging") cpu_staging = value;
            else throw std::runtime_error("Unknown option: " + key);
        }
        if (iterations < 1 || warmup < 0 || threads < 0 || cpu_staging < 0 || cpu_staging > 1)
            throw std::runtime_error("Invalid benchmark counts");
        cv::setNumThreads(threads);
        cv::Mat image = cv::imread(image_path.string());
        if (image.empty()) throw std::runtime_error("Cannot read image: " + image_path.string());
        if (!image.isContinuous()) image = image.clone();
        TrtEngine engine;
        if (!engine.loadEngine(engine_path.string())) return 1;
        const auto stream = engine.stream();
        const int n = engine.outputSize() / 6;
        if (n > 1024) throw std::runtime_error("CUDA postprocessor currently supports N<=1024");
        const size_t input_bytes = size_t(engine.inputSize()) * sizeof(float);
        const size_t output_bytes = size_t(engine.outputSize()) * sizeof(float);
        const size_t raw_bytes = size_t(image.cols) * image.rows * 3;
        static_assert(alignof(Detection) <= alignof(int), "Packet alignment");
        const size_t packet_bytes = sizeof(int) + n * sizeof(Detection);
        Memory host_input(input_bytes, true), host_raw(raw_bytes, true), host_output(output_bytes, true);
        std::vector<float> cpu_workspace(cpu_staging ? engine.inputSize() : 0);
        Memory device_packet(packet_bytes, false), host_packet(packet_bytes, true);
        check(cudaMemset(device_packet.ptr, 0, packet_bytes));
        if (!engine.buffers_info().ensureRawCapacity(raw_bytes)) throw std::runtime_error("Raw allocation failed");
        CudaPreprocessor preprocessor;
        check(preprocessor.configureResize(image.cols, image.rows, engine.inputWidth(), engine.inputHeight()));
        auto* device_detections = reinterpret_cast<Detection*>(static_cast<char*>(device_packet.ptr) + sizeof(int));
        auto* host_detections = reinterpret_cast<Detection*>(static_cast<char*>(host_packet.ptr) + sizeof(int));
        OpenCVPreprocessWorkspace resize_workspace;
        resize_workspace.create(engine.inputWidth(), engine.inputHeight());
        std::vector<Detection> detections;
        detections.reserve(n);
        Events events;
        StreamCompletionGuard completion{stream};

        auto run = [&](bool gpu) {
            Sample s;
            const auto begin = Clock::now();
            if (gpu) std::memcpy(host_raw.ptr, image.data, raw_bytes);
            else {
                auto* destination = cpu_staging ? cpu_workspace.data() : static_cast<float*>(host_input.ptr);
                Preprocessor::preprocessOpenCV(image, destination,
                    engine.inputWidth(), engine.inputHeight(), resize_workspace);
                if (cpu_staging) std::memcpy(host_input.ptr, cpu_workspace.data(), input_bytes);
            }
            const auto prepared = Clock::now();
            s.cpu_prepare = ms(begin, prepared);
            events.record(0, stream);
            if (gpu) check(cudaMemcpyAsync(engine.buffers_info().raw_device(), host_raw.ptr,
                raw_bytes, cudaMemcpyHostToDevice, stream));
            else check(cudaMemcpyAsync(engine.buffers_info().input_device(), host_input.ptr,
                input_bytes, cudaMemcpyHostToDevice, stream));
            events.record(1, stream);
            if (gpu) check(preprocessor.resizeNormalize(
                static_cast<const unsigned char*>(engine.buffers_info().raw_device()), image.cols * 3,
                static_cast<float*>(engine.buffers_info().input_device()), stream));
            events.record(2, stream);
            if (!engine.enqueue()) throw std::runtime_error("TensorRT enqueue failed");
            events.record(3, stream);
            if (gpu) check(CudaPostprocessor::decode(static_cast<const float*>(engine.buffers_info().output_device()),
                n, 0.25f, image.cols, image.rows, 80, device_detections,
                static_cast<int*>(device_packet.ptr), stream));
            events.record(4, stream);
            if (gpu) check(cudaMemcpyAsync(host_packet.ptr, device_packet.ptr,
                packet_bytes, cudaMemcpyDeviceToHost, stream));
            else check(cudaMemcpyAsync(host_output.ptr, engine.buffers_info().output_device(),
                output_bytes, cudaMemcpyDeviceToHost, stream));
            events.record(5, stream);
            // Completion is established by this event, never by an assumed delay.
            check(cudaEventSynchronize(events.e[5]));
            const auto post_begin = Clock::now();
            if (gpu) {
                const int count = *static_cast<const int*>(host_packet.ptr);
                if (count < 0 || count > n) throw std::runtime_error("Invalid GPU detection count");
                detections.resize(count);
                std::copy_n(host_detections, count, detections.begin());
            } else Postprocessor::decodeInto(static_cast<const float*>(host_output.ptr), n,
                0.25f, image.cols, image.rows, 80, detections);
            const auto end = Clock::now();
            s.cpu_post = ms(post_begin, end);
            s.total = ms(begin, end);
            // Read event timings after the wall-clock interval to avoid charging
            // statistics collection to end-to-end latency.
            s.h2d = events.elapsed(0, 1); s.gpu_pre = gpu ? events.elapsed(1, 2) : 0;
            s.infer = events.elapsed(2, 3); s.gpu_post = gpu ? events.elapsed(3, 4) : 0;
            s.d2h = events.elapsed(4, 5);
            return s;
        };

        // Correctness is checked before measuring warm steady-state latency.
        run(false);
        std::vector<float> cpu_input(engine.inputSize()), cpu_output(engine.outputSize());
        check(cudaMemcpy(cpu_input.data(), engine.buffers_info().input_device(), input_bytes, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(cpu_output.data(), engine.buffers_info().output_device(), output_bytes, cudaMemcpyDeviceToHost));
        run(true);
        std::vector<float> gpu_input(engine.inputSize()), gpu_output(engine.outputSize());
        check(cudaMemcpy(gpu_input.data(), engine.buffers_info().input_device(), input_bytes, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(gpu_output.data(), engine.buffers_info().output_device(), output_bytes, cudaMemcpyDeviceToHost));
        const float input_error = difference(cpu_input, gpu_input), output_error = difference(cpu_output, gpu_output);
        std::vector<Detection> reference;
        Postprocessor::decodeInto(gpu_output.data(), n, 0.25f, image.cols, image.rows, 80, reference);
        if (reference.size() != detections.size()) throw std::runtime_error("CPU/CUDA decode counts differ");
        float box_error = 0;
        for (size_t i = 0; i < reference.size(); ++i) {
            const auto& a = reference[i]; const auto& b = detections[i];
            if (a.class_id != b.class_id || a.confidence != b.confidence)
                throw std::runtime_error("CPU/CUDA decode order, class or score differs");
            box_error = std::max({box_error, std::abs(a.x1-b.x1), std::abs(a.y1-b.y1),
                std::abs(a.x2-b.x2), std::abs(a.y2-b.y2)});
        }
        std::cout << "Checks: input max=" << input_error << ", raw output max=" << output_error
                  << ", decode box max=" << box_error << ", detections=" << detections.size() << '\n';
        if (input_error > 1e-7f || output_error > 1e-5f || box_error > 1e-4f)
            throw std::runtime_error("Numerical parity check failed; benchmark stopped");
        for (int i = 0; i < warmup; ++i) { run(false); run(true); }
        std::vector<Sample> cpu, gpu;
        cpu.reserve(iterations); gpu.reserve(iterations);
        for (int i = 0; i < iterations; ++i) {
            // Alternate the first mode to reduce ordering/clock drift bias.
            if (i % 2) { gpu.push_back(run(true)); cpu.push_back(run(false)); }
            else { cpu.push_back(run(false)); gpu.push_back(run(true)); }
        }
        if (!prefix.parent_path().empty()) fs::create_directories(prefix.parent_path());
        std::ofstream csv(prefix.string() + ".samples.csv");
        csv << "mode,iteration";
        for (const auto& f : fields) csv << ',' << f.first << "_ms";
        csv << '\n' << std::setprecision(9);
        for (int mode = 0; mode < 2; ++mode) {
            const auto& samples = mode ? gpu : cpu;
            for (size_t i = 0; i < samples.size(); ++i) {
                csv << (mode ? "cuda" : "cpu") << ',' << i;
                for (const auto& f : fields) csv << ',' << samples[i].*(f.second);
                csv << '\n';
            }
        }
        std::ofstream json(prefix.string() + ".summary.json");
        json << std::setprecision(9) << "{\n\"engine\":" << std::quoted(engine_path.string())
             << ",\n\"image\":" << std::quoted(image_path.string())
             << ",\n\"width\":" << image.cols << ",\"height\":" << image.rows
             << ",\"opencv_version\":" << std::quoted(CV_VERSION)
             << ",\"opencv_threads\":" << cv::getNumThreads()
             << ",\"cpu_preprocess\":\"OpenCV resize+cvtColor+convertTo+split\""
             << ",\"cpu_staging\":" << cpu_staging
             << ",\"iterations_per_mode\":" << iterations << ",\"warmup_per_mode\":" << warmup
             << ",\"lanes\":1,\"h2d_cpu_bytes\":" << input_bytes << ",\"h2d_cuda_bytes\":" << raw_bytes
             << ",\"d2h_cpu_bytes\":" << output_bytes << ",\"d2h_cuda_bytes\":" << packet_bytes
             << ",\n\"input_max_abs\":" << input_error << ",\"output_max_abs\":" << output_error
             << ",\"postprocess_box_max_abs\":" << box_error << ",\"detections\":" << detections.size()
             << ",\n\"cpu\":";
        writeStats(json, cpu); json << ",\n\"cuda\":"; writeStats(json, gpu); json << "\n}\n";
        if (!csv || !json) throw std::runtime_error("Cannot write benchmark results");
        std::cout << "Single lane; OpenCV threads=" << cv::getNumThreads()
                  << "; warmup=" << warmup << "; paired iterations=" << iterations << '\n'
                  << std::fixed << std::setprecision(4) << std::left
                  << std::setw(18) << "Mean ms" << std::setw(14) << "CPU/OpenCV" << "CUDA\n";
        for (const auto& f : fields) std::cout << std::setw(18) << f.first
            << std::setw(14) << stats(cpu, f.second).mean << stats(gpu, f.second).mean << '\n';
        const auto a = stats(cpu, &Sample::total), b = stats(gpu, &Sample::total);
        std::cout << "E2E median: " << a.median << " / " << b.median << " ms\n"
                  << "E2E p95: " << a.p95 << " / " << b.p95 << " ms\n"
                  << "Saved: " << prefix << ".{samples.csv,summary.json}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n'; return 1;
    }
}
