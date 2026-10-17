import numpy as np 
import cv2
import onnxruntime as ort
from pathlib import Path

ONNX_PATH = "/home/smisei/project/RTDETR-ONNX-TensorRT/weights/rtdetr-l.onnx"
IMAGE_PATH = "/home/smisei/project/RTDETR-ONNX-TensorRT/ultralytics/coco/images/val2017/000000057027.jpg"

image = cv2.imread(IMAGE_PATH)               # BGR
resized = cv2.resize(image, (640, 640))
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
x = rgb.transpose(2, 0, 1)[None]              # [1, 3, 640, 640]
x = np.ascontiguousarray(x, dtype=np.float32) / 255.0

ort.preload_dlls(directory="")
session = ort.InferenceSession(ONNX_PATH,providers=["CUDAExecutionProvider"])
print("Session providers:", session.get_providers())
if "CUDAExecutionProvider" not in session.get_providers():
    raise RuntimeError("CUDA 后端加载失败，请检查上方日志")
input_name = session.get_inputs()[0].name
outputs = session.run(None,{input_name:x})
# outputs[0]: [1, 300, 6]
pred = outputs[0][0]
pred = pred[pred[:, 4] >= 0.25]

# 归一化 cxcywh → 原图像素 xyxy
h, w = image.shape[:2]
boxes = np.empty_like(pred[:, :4])
boxes[:, 0] = (pred[:, 0] - pred[:, 2] / 2) * w
boxes[:, 1] = (pred[:, 1] - pred[:, 3] / 2) * h
boxes[:, 2] = (pred[:, 0] + pred[:, 2] / 2) * w
boxes[:, 3] = (pred[:, 1] + pred[:, 3] / 2) * h

scores = pred[:, 4]
class_ids = pred[:, 5].astype(np.int64)

for box,score,cls in zip(
    boxes,
    scores,
    class_ids
):
    x1,y1,x2,y2=box.astype(int)
    cv2.rectangle(
        image,
        (x1,y1),
        (x2,y2),
        (0,255,0),
        2
    )


    cv2.putText(
        image,
        f"{cls}:{score:.2f}",
        (x1,y1-5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0,255,0),
        2
    )
cv2.imwrite(str("/home/smisei/project/RTDETR-ONNX-TensorRT/result.jpg"), image)
print(boxes)
print(scores)
print(class_ids)