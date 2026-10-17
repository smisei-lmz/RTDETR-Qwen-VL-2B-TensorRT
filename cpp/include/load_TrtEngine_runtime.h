#pragma once

#include <NvInfer.h>

#include <memory>
#include <string>
#include <vector>

#include "Logger.h"

// Owns one shared model. Destroy all lane contexts before this object.
class SharedTrtEngine {
public:
    SharedTrtEngine();
    ~SharedTrtEngine();
    SharedTrtEngine(const SharedTrtEngine&) = delete;
    SharedTrtEngine& operator=(const SharedTrtEngine&) = delete;

    // Load once. Failure leaves the object unchanged and permits a retry.
    bool loadEngine(const std::string& path);

    int inputSize() const noexcept { return input_size_; }
    int outputSize() const noexcept { return output_size_; }
    int inputHeight() const noexcept { return input_height_; }
    int inputWidth() const noexcept { return input_width_; }
    const std::string& inputName() const noexcept { return input_name_; }
    const std::string& outputName() const noexcept { return output_name_; }

    // Borrowed pointer for creating lane contexts; null until loading succeeds.
    nvinfer1::ICudaEngine* engine() const noexcept { return engine_.get(); }

private:
    static std::vector<char> loadFile(const std::string& path);

    int input_size_{0};
    int output_size_{0};
    int input_width_{0};
    int input_height_{0};
    std::string input_name_;
    std::string output_name_;

    // Reverse destruction order: engine, runtime, logger.
    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
};
