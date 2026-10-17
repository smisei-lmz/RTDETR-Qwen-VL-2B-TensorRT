#include "Postprocess.h"
#include <cmath>
#include <stdexcept>

std::vector<Detection> Postprocessor::decode(const float* output, int num_boxes,
    float conf_threshold, int img_w, int img_h, int num_classes) {
    std::vector<Detection> results;
    decodeInto(output, num_boxes, conf_threshold, img_w, img_h, num_classes, results);
    return results;
}

void Postprocessor::decodeInto(const float* output, int num_boxes,
    float conf_threshold, int img_w, int img_h, int num_classes,
    std::vector<Detection>& results) {
    if (!output || num_boxes <= 0 || img_w <= 0 || img_h <= 0)
        throw std::invalid_argument("Invalid output tensor or image size");
    results.clear();
    // This exported RT-DETR: [1, N, 6], normalized cx, cy, w, h, score, class_id.
    // Top-K is inside the graph; there is no external argmax, sigmoid or NMS.
    for (int i = 0; i < num_boxes; ++i) {
        const float* row = output + i * 6;
        bool finite = true;
        for (int j = 0; j < 6; ++j) finite &= std::isfinite(row[j]);
        if (!finite || row[4] < conf_threshold || row[2] <= 0 || row[3] <= 0) continue;
        if (row[5] < 0 || row[5] >= num_classes) continue;
        const int class_id = static_cast<int>(std::lround(row[5]));
        if (class_id < 0 || class_id >= num_classes || std::abs(row[5] - class_id) > 1e-3f) continue;
        Detection det;
        det.class_id = class_id;
        det.confidence = row[4];
        det.x1 = (row[0] - row[2] * 0.5f) * img_w;
        det.y1 = (row[1] - row[3] * 0.5f) * img_h;
        det.x2 = (row[0] + row[2] * 0.5f) * img_w;
        det.y2 = (row[1] + row[3] * 0.5f) * img_h;
        // Preserve floating coordinates; clip only for drawing.
        results.push_back(det);
    }
}
