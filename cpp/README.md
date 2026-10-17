# RT-DETR TensorRT C++

## 单 lane 前后处理比较

已新增完整 CPU/OpenCV 与 CUDA 两条链路的比较程序：

```bash
cmake --build cpp/build-trt10 -j 4
./cpp/build-trt10/trt_benchmark
```

每帧重新执行前后处理和传输，以完成事件确认结果可读。详细方法、条件和实测结果见
[BENCHMARK.md](BENCHMARK.md)。原 `trt_example` 是下面描述的单图流程。

从相邻 `Yolov11-ONNX-TensorRT/cpp` 复用 TrtEngine、BufferManager、CUDA
颜色/布局转换、Logger、NVTX、计时与绘图流程。

## 编译与运行

在 `RTDETR-ONNX-TensorRT` 项目根目录：

```bash
cmake -S cpp -B cpp/build-trt10 -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build-trt10 -j 4
./cpp/build-trt10/trt_example
```

默认模型：`models/rtdetr-l-int8-box-add6-fp32.engine`。
默认图片：`ultralytics/coco/images/val2017/000000147740.jpg`。
默认结果：`results/rtdetr_result.jpg`，通过 imwrite 保存。

指定模型、图片和保存位置：

```bash
./cpp/build-trt10/trt_example \
  models/rtdetr-l-int8-box-add6-fp32.engine \
  ultralytics/coco/images/val2017/000000057027.jpg \
  results/rtdetr_result.jpg
```

`--repeat 100` 复用已经预处理好的输入，统计推理及输出复制的平均耗时。
默认 warmup=10、repeat=1；这不是包含图像读取、预处理的端到端 FPS。

## 相对 YOLO 修改的位置

| 文件 | 修改 |
| --- | --- |
| src/Preprocess.cpp | LetterBox 改成 OpenCV INTER_LINEAR 直接 resize |
| src/CudaPreprocess.cu | 沿用 CUDA BGR→RGB、HWC→CHW、FP32/255，使用推理 stream |
| src/Postprocess.cpp | 读取 [1,N,6]，直接取 score/class_id，归一化框乘原图宽高；不做 NMS |
| src/TrtEngine.cpp | 沿用 enqueueV3 和计时；检查静态 batch=1、FP32 线性 I/O、输出最后维度=6 |
| main.cpp | 改模型、图片路径和 decode 调用；支持参数、原始张量/框导出 |
| CMakeLists.txt | 匹配当前 TensorRT 10.13.3 engine，不再链接 TensorRT 11 |

当前导出的模型输入 `[1,3,640,640]`、输出 `[1,300,6]`，两者都是 FP32。
输出每行是归一化 `[cx,cy,w,h,score,class_id]`，Top-K 已在图中。
不适用旧版 `[1,300,84]` 或 YOLO `[1,84,8400]`，形状不匹配会拒绝加载。

默认预处理：**CPU OpenCV 在 uint8 上 resize，然后 H2D，CUDA 转 RGB/CHW/FP32 并除以255**。
这样保留 Python cv2.resize 的插值和 uint8 舍入，不使用原 YOLO 的 LetterBox/fused resize。
`--cpu-preprocess` 可切换为整个预处理在 CPU 完成后上传 FP32。
框保留浮点，绘图时才裁剪和取整。

## TensorRT 与 CUDA

- 头文件：项目内 `third_party/TensorRT-10.13.3/include`，来自 NVIDIA 官方 v10.13.3（10.13.3.9）。
- 动态库：项目 `.venv-trt10/lib/python3.10/site-packages/tensorrt_libs/libnvinfer.so.10`。
- 默认沿用本机 `/usr/local/cuda-13.4/bin/nvcc`，按本机 GPU 编译 CUDA kernel。
- 使用单独的 build-trt10，避免旧 build 中的 TensorRT 11 CMake 缓存。
- 也可通过 `-DTENSORRT_ROOT=/path/to/TensorRT-10.13.3` 指定完整 SDK；
  或分别设置 `TENSORRT_INCLUDE_DIR`、`TENSORRT_LIBRARY`。

## Python 数值对齐

```bash
./cpp/build-trt10/trt_example \
  models/rtdetr-l-int8-box-add6-fp32.engine \
  ultralytics/coco/images/val2017/000000147740.jpg \
  results/cpp_result.jpg --dump results/cpp_parity/gpu

.venv-trt10/bin/python cpp/verify.py \
  --engine models/rtdetr-l-int8-box-add6-fp32.engine \
  --image ultralytics/coco/images/val2017/000000147740.jpg \
  --prefix results/cpp_parity/gpu
```

`--dump` 输出 `.input.bin`、`.output.bin`（连续 float32）和 `.detections.csv`。
verify.py 分别比较预处理输入、相同输入的原始 engine 输出、Python 预处理的输出、坐标解码。

本次在 RTX 5060、保护六处 Add 的 TensorRT 10.13.3.9 engine 上测试上述图片：

- C++ 默认预处理与 Python 输入最大绝对误差：0。
- C++ 与 Python 原始输出最大绝对误差：0。
- 15 个检测框一致；CSV 坐标保存后最大绝对误差约 4.71e-7 像素。
- CPU 与默认 CUDA 转换路径的输入、原始输出逐元素一致。

这是单图数值对齐结果，尚未执行 C++ 完整验证集 mAP。
