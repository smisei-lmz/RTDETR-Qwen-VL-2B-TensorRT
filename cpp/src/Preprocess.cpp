#include "Preprocess.h"
#include <stdexcept>

void Preprocessor::preprocessOpenCV(const cv::Mat& image, float* output, int width, int height,
                                   OpenCVPreprocessWorkspace& workspace) {
    if (!output || image.empty() || image.type() != CV_8UC3 || width <= 0 || height <= 0)
        throw std::invalid_argument("Invalid OpenCV preprocessing input");
    workspace.create(width, height); // Mat::create reuses correctly sized storage.
    cv::resize(image, workspace.resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(workspace.resized, workspace.rgb, cv::COLOR_BGR2RGB);
    workspace.rgb.convertTo(workspace.fp32, CV_32FC3, 1.0 / 255.0);
    const int area = width * height;
    cv::Mat channels[] = {
        cv::Mat(height, width, CV_32F, output),
        cv::Mat(height, width, CV_32F, output + area),
        cv::Mat(height, width, CV_32F, output + 2 * area)};
    cv::split(workspace.fp32, channels);
}

cv::Mat Preprocessor::resize(const cv::Mat& image, int width, int height) {
    if (image.empty() || image.type() != CV_8UC3 || width <= 0 || height <= 0)
        throw std::invalid_argument("Expected a non-empty BGR uint8 image and positive input dimensions");
    cv::Mat resized;
    // Match Python cv2.resize on uint8, before conversion to floating point.
    cv::resize(image, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    return resized;
}

void Preprocessor::preprocess(const cv::Mat& image, float* output, int width, int height,
                             cv::Mat* resize_workspace) {
    if (!output) throw std::invalid_argument("Null input buffer");
    if (image.empty() || image.type() != CV_8UC3 || width <= 0 || height <= 0)
        throw std::invalid_argument("Invalid preprocessing image/dimensions");
    cv::Mat temporary;
    cv::Mat& img = resize_workspace ? *resize_workspace : temporary;
    cv::resize(image, img, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    const int area = width * height;
    for (int y = 0; y < height; ++y) {
        const auto* row = img.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; ++x) {
            const int offset = y * width + x;
            output[offset] = row[x][2] / 255.0f;
            output[area + offset] = row[x][1] / 255.0f;
            output[2 * area + offset] = row[x][0] / 255.0f;
        }
    }
}
