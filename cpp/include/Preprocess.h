#pragma once 

#include "Types.h"
#include <opencv2/opencv.hpp>

struct OpenCVPreprocessWorkspace {
    cv::Mat resized, rgb, fp32;
    void create(int width, int height) {
        resized.create(height, width, CV_8UC3);
        rgb.create(height, width, CV_8UC3);
        fp32.create(height, width, CV_32FC3);
    }
};

class Preprocessor{
public:
    static cv::Mat resize(const cv::Mat& image, int width, int height);
    static void preprocess(const cv::Mat& image, float* output, int width, int height,
                           cv::Mat* resize_workspace = nullptr);
    static void preprocessOpenCV(const cv::Mat& image, float* output, int width, int height,
                                 OpenCVPreprocessWorkspace& workspace);
};
