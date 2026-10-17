#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <opencv2/opencv.hpp>
#include "TrtEngine.h"
#include "Preprocess.h"
#include "CudaPreprocess.h"
#include "Postprocess.h"
#include "Labels.h"
#include "nvtx_range.h"

namespace fs = std::filesystem;
namespace {
void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void saveFloats(const fs::path& path, const std::vector<float>& values) {
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::binary);
    if (!file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float)))
        throw std::runtime_error("Cannot save " + path.string());
}
}

int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: trt_example [engine image result.jpg] [--cpu-preprocess] [--dump PREFIX] [--repeat N] [--warmup N]\n"
                      << "Default: OpenCV uint8 resize -> GPU RGB/CHW/normalize -> TensorRT -> CPU decode.\n";
            return 0;
        }
        const fs::path root(RTDETR_PROJECT_DIR);
        fs::path engine_path = root / "models/rtdetr-l-int8-box-add6-fp32.engine";
        fs::path image_path = root / "ultralytics/coco/images/val2017/000000147740.jpg";
        fs::path result_path = root / "results/rtdetr_result.jpg";
        fs::path dump_prefix;
        bool cpu_preprocess = false;
        int repeat = 1, warmup = 10;
        int arg = 1;
        if (arg < argc && std::string(argv[arg]).rfind("--", 0) != 0) {
            if (argc < 4) throw std::runtime_error("Provide engine, image and result paths; see --help");
            engine_path = argv[arg++]; image_path = argv[arg++]; result_path = argv[arg++];
        }
        for (; arg < argc; ++arg) {
            const std::string option(argv[arg]);
            if (option == "--cpu-preprocess") cpu_preprocess = true;
            else if (option == "--dump" && arg + 1 < argc) dump_prefix = argv[++arg];
            else if (option == "--repeat" && arg + 1 < argc) repeat = std::stoi(argv[++arg]);
            else if (option == "--warmup" && arg + 1 < argc) warmup = std::stoi(argv[++arg]);
            else throw std::runtime_error("Unknown/incomplete option: " + option);
        }
        if (repeat < 1 || warmup < 0) throw std::runtime_error("repeat >= 1 and warmup >= 0 required");
        cv::Mat image = cv::imread(image_path.string());
        if (image.empty()) throw std::runtime_error("Cannot read image: " + image_path.string());
        std::cout << "TensorRT headers: " << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR
                  << '.' << NV_TENSORRT_PATCH << "; runtime version: " << getInferLibVersion() << '\n';
        std::cout << "Image: " << image.cols << 'x' << image.rows << '\n';
        TrtEngine engine;
        if (!engine.loadEngine(engine_path.string())) return 1;
        std::vector<float> input, output(engine.outputSize());
        const auto preprocess_start = std::chrono::steady_clock::now();
        if (cpu_preprocess) {
            input.resize(engine.inputSize());
            Preprocessor::preprocess(image, input.data(), engine.inputWidth(), engine.inputHeight());
            checkCuda(cudaMemcpyAsync(engine.buffers_info().input_device(), input.data(),
                input.size() * sizeof(float), cudaMemcpyHostToDevice, engine.stream()));
        } else {
            // The YOLO CUDA color/layout/normalization path, without LetterBox.
            NvtxRange range("rtdetr_preprocess");
            const cv::Mat resized = Preprocessor::resize(image, engine.inputWidth(), engine.inputHeight());
            const std::size_t row_bytes = static_cast<std::size_t>(resized.cols) * 3;
            if (!engine.buffers_info().ensureRawCapacity(row_bytes * resized.rows))
                throw std::runtime_error("Cannot allocate raw image GPU buffer");
            checkCuda(cudaMemcpy2DAsync(engine.buffers_info().raw_device(), row_bytes,
                resized.data, resized.step, row_bytes, resized.rows, cudaMemcpyHostToDevice, engine.stream()));
            CudaPreprocessor preprocessor;
            checkCuda(preprocessor.CudaBGR2RGB_HWC2CHW_Normalize(
                static_cast<const unsigned char*>(engine.buffers_info().raw_device()),
                static_cast<float*>(engine.buffers_info().input_device()),
                engine.inputWidth(), engine.inputHeight(), engine.stream()));
            // Keep resized's pageable host storage alive until its copy has completed.
            checkCuda(cudaStreamSynchronize(engine.stream()));
        }
        checkCuda(cudaStreamSynchronize(engine.stream()));
        const double preprocess_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - preprocess_start).count();
        for (int i = 0; i < warmup; ++i)
            if (!engine.infer_gpubuffer(output.data())) return 1;
        InferenceTiming average;
        for (int i = 0; i < repeat; ++i) {
            InferenceTiming current;
            if (!engine.infer_gpubuffer(output.data(), &current)) return 1;
            average.execute_ms += current.execute_ms;
            average.d2h_ms += current.d2h_ms;
            average.gpu_total_ms += current.gpu_total_ms;
            average.sync_wait_ms += current.sync_wait_ms;
        }
        const auto post_start = std::chrono::steady_clock::now();
        const auto detections = Postprocessor::decode(output.data(), engine.outputSize() / 6,
            0.25f, image.cols, image.rows, static_cast<int>(kCocoLabels.size()));
        const double post_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - post_start).count();
        std::cout << std::fixed << std::setprecision(3)
                  << "Preprocess including H2D: " << preprocess_ms << " ms\n"
                  << "Average over " << repeat << " runs (preprocessing excluded):\n"
                  << "  TensorRT: " << average.execute_ms / repeat << " ms\n"
                  << "  D2H: " << average.d2h_ms / repeat << " ms\n"
                  << "  GPU inference + D2H: " << average.gpu_total_ms / repeat << " ms\n"
                  << "  Host sync wait: " << average.sync_wait_ms / repeat << " ms\n"
                  << "CPU postprocess: " << post_ms << " ms\n"
                  << "Detections: " << detections.size() << '\n';
        if (!dump_prefix.empty()) {
            input.resize(engine.inputSize());
            checkCuda(cudaMemcpy(input.data(), engine.buffers_info().input_device(),
                input.size() * sizeof(float), cudaMemcpyDeviceToHost));
            saveFloats(dump_prefix.string() + ".input.bin", input);
            saveFloats(dump_prefix.string() + ".output.bin", output);
            std::ofstream csv(dump_prefix.string() + ".detections.csv");
            csv << "class_id,score,x1,y1,x2,y2\n" << std::setprecision(9);
            for (const auto& d : detections)
                csv << d.class_id << ',' << d.confidence << ',' << d.x1 << ',' << d.y1
                    << ',' << d.x2 << ',' << d.y2 << '\n';
            if (!csv) throw std::runtime_error("Cannot save detections CSV");
        }
        for (const auto& d : detections) {
            const int x1 = static_cast<int>(std::clamp(d.x1, 0.0f, float(image.cols - 1)));
            const int y1 = static_cast<int>(std::clamp(d.y1, 0.0f, float(image.rows - 1)));
            const int x2 = static_cast<int>(std::clamp(d.x2, 0.0f, float(image.cols - 1)));
            const int y2 = static_cast<int>(std::clamp(d.y2, 0.0f, float(image.rows - 1)));
            cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
            const std::string text = std::string(kCocoLabels[d.class_id]) + ":" + std::to_string(d.confidence).substr(0, 4);
            cv::putText(image, text, cv::Point(x1, std::max(15, y1 - 5)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }
        if (!result_path.parent_path().empty()) fs::create_directories(result_path.parent_path());
        if (!cv::imwrite(result_path.string(), image)) throw std::runtime_error("Cannot save result image");
        std::cout << "Saved: " << result_path << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
