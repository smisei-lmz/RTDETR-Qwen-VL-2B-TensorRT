#pragma once 
#include <vector>
#include "Types.h"

class Postprocessor{
public:
    static std::vector<Detection> decode(const float* output, int num_boxes,
        float conf_threshold, int img_w, int img_h, int num_classes = 80);
    static void decodeInto(const float* output, int num_boxes, float conf_threshold,
        int img_w, int img_h, int num_classes, std::vector<Detection>& results);
};

