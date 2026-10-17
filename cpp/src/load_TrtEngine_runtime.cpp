#include "load_TrtEngine_runtime.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

SharedTrtEngine::SharedTrtEngine() = default;
SharedTrtEngine::~SharedTrtEngine() = default;

std::vector<char> SharedTrtEngine::loadFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open engine: " + path);

    const auto end = file.tellg();
    if (end <= 0) throw std::runtime_error("Engine file is empty or unreadable: " + path);
    const auto size = static_cast<std::size_t>(end);
    std::vector<char> data(size);
    file.seekg(0, std::ios::beg);
    if (!file.read(data.data(), static_cast<std::streamsize>(size)))
        throw std::runtime_error("Cannot read engine: " + path);
    return data;
}

bool SharedTrtEngine::loadEngine(const std::string& path) {
    if (engine_) {
        std::cerr << "Engine is already loaded\n";
        return false;
    }

    try {
        auto data = loadFile(path);
        std::unique_ptr<nvinfer1::IRuntime> runtime{
            nvinfer1::createInferRuntime(logger_)
        };
        if (!runtime) {
            std::cerr << "Cannot create TensorRT runtime\n";
            return false;
        }

        std::unique_ptr<nvinfer1::ICudaEngine> engine{
            runtime->deserializeCudaEngine(data.data(), data.size())
        };
        if (!engine) {
            std::cerr << "Cannot deserialize TensorRT engine: " << path << '\n';
            return false;
        }

        if (engine->getNbIOTensors() != 2) {
            std::cerr << "Expected exactly one input and one output\n";
            return false;
        }

        // Validate into local variables so failure cannot leave partial state.
        std::string input_name, output_name;
        int input_size = 0, output_size = 0;
        int input_width = 0, input_height = 0;
        for (int i = 0; i < engine->getNbIOTensors(); ++i) {
            const char* name = engine->getIOTensorName(i);
            if (!name || !*name) {
                std::cerr << "Invalid I/O tensor name\n";
                return false;
            }
            const auto mode = engine->getTensorIOMode(name);
            const auto shape = engine->getTensorShape(name);

            if (engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT ||
                engine->getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR ||
                engine->getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE) {
                std::cerr << "Expected FP32 linear device I/O: " << name << '\n';
                return false;
            }

            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                if (!input_name.empty() || shape.nbDims != 4 ||
                    shape.d[0] != 1 || shape.d[1] != 3) {
                    std::cerr << "Expected one static [1,3,H,W] input\n";
                    return false;
                }
            } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
                if (!output_name.empty() || shape.nbDims != 3 ||
                    shape.d[0] != 1 || shape.d[2] != 6) {
                    std::cerr << "Expected one static [1,N,6] RT-DETR output\n";
                    return false;
                }
            } else {
                std::cerr << "Unsupported tensor I/O mode: " << name << '\n';
                return false;
            }

            std::int64_t size = 1;
            for (int j = 0; j < shape.nbDims; ++j) {
                if (shape.d[j] <= 0 ||
                    size > std::numeric_limits<int>::max() / shape.d[j]) {
                    std::cerr << "Unsupported dynamic or oversized tensor: " << name << '\n';
                    return false;
                }
                size *= shape.d[j];
            }

            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                input_name = name;
                input_size = static_cast<int>(size);
                input_height = static_cast<int>(shape.d[2]);
                input_width = static_cast<int>(shape.d[3]);
            } else {
                output_name = name;
                output_size = static_cast<int>(size);
            }
        }

        if (input_name.empty() || output_name.empty()) {
            std::cerr << "Expected exactly one input and one output\n";
            return false;
        }

        // Publish the model only after all checks pass.
        runtime_ = std::move(runtime);
        engine_ = std::move(engine);
        input_name_ = std::move(input_name);
        output_name_ = std::move(output_name);
        input_size_ = input_size;
        output_size_ = output_size;
        input_width_ = input_width;
        input_height_ = input_height;
        return true;
    } catch (const std::exception& error) {
        std::cerr << "Cannot load engine: " << error.what() << '\n';
        return false;
    }
}
