# RT-DETR TensorRT C++

## DETR 收尾验收

当前主线固定为 **Ultralytics RT-DETR-L**，PC 默认采用 1 lane。
统一配置见 [config/detr_deployment.json](config/detr_deployment.json)。
验收会先按同一套 C++ CUDA 前后处理比较 FP32、混合 FP16 与 INT8，
再把通过精度预算的部署引擎写入 `results/detr_closeout_20261019/deployment.lock.json`。
历史 INT8 性能结果保留原模型和哈希，不与新引擎结果合并。

使用现有 TensorRT 10 环境，另安装 `pycocotools==2.0.11`。
旧的 `rtdetr-l-fp32.engine`、`rtdetr-l-fp32mix16.engine` 与该环境不兼容，
因此从本地 ONNX 生成带 `-trt10` 后缀的新文件，不覆盖旧引擎：

```bash
.venv-trt10/bin/python detr_export_engine.py \
  --onnx models/rtdetr-l-fp32.onnx --output models/rtdetr-l-fp32-trt10.engine
.venv-trt10/bin/python detr_export_engine.py \
  --onnx models/rtdetr-l-fp32mix16.onnx --output models/rtdetr-l-fp32mix16-trt10.engine
cmake --build cpp/build-trt10 -j 4
.venv-trt10/bin/python cpp/closeout.py evaluate
.venv-trt10/bin/python cpp/closeout.py checks
.venv-trt10/bin/python cpp/closeout.py stability --seconds 5
.venv-trt10/bin/python cpp/analyze_closeout.py --waive-long-stability
```

`evaluate` 对本地可用且在 COCO JSON 中有标注的所有图片运行实际 C++ lane，
用 pycocotools 的 bbox COCOeval 统计 mAP；本机是 4500/5000 张。
评估阈值 0.001、COCO maxDets=100，导出时将框裁到图像范围，不增加 NMS。
混合 FP16 相对同后端 FP32 最多下降 1 个百分点时优先用于部署，否则回退 FP32。
这与运行时默认置信阈值 0.25 分开配置。

`checks` 在选定引擎上执行 CPU/CUDA 对照与五轮 1/2 lane 性能测试，
并保存该引擎及 INT8 候选的 lane 正确性检查。`analyze_closeout.py`
核对配置/引擎哈希、逐帧身份与验收条件，输出 `closeout.json` 和 `overview.png`。

本次按用户要求取消 30 分钟长测，已保存约 190 秒的中断采样；它不算完成长测。
`stability` 默认执行 5 秒短检查，在选定引擎上交替输入两张预加载图片，
每帧检查 ID、检测值重复一致性和完成状态；记录逐帧延迟、进程 RSS、
全设备可用显存，并在所有结果取走后销毁 lane、采样资源回收情况。
这验证固定输入的推理与正常回收，不包含实际视频输入节拍、ROS 或 VLM。
分析时用 `--waive-long-stability` 明确记录取消长期验收，不将其标为通过。
原计划 1800 秒预算仍在冻结配置中；以后需要长测时显式传 `--seconds 1800`，
完成后分析可不加豁免参数。
大体积原始预测与日志位于 `results/detr_closeout_20261019/raw/`，已被 Git 忽略；
轻量汇总、模型/数据哈希和最终配置保留在上一级目录。

可用 `--count 2 --output results/detr_closeout_smoke_20261019` 做独立的小规模流程检查；
短时稳定性检查加 `--seconds 2`，它不会被标记为满足 30 分钟验收。

底层 `trt_lane_validate --help` 提供单独的数据集与稳定性入口。
`LaneConfig` 可设置置信阈值和类别数；原有 `submit/query/collect` 调用方式保持兼容。

## 定量分析与简历依据

完整测量条件、1/2 lane 对比、Nsight Systems 诊断和简历表述见
[OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md)。

```bash
./cpp/build-trt10/trt_lane_benchmark --frames 500 --warmup 30 --rounds 5 --lanes both --poll-us 50
```

单独测INT8速度可直接指定引擎，不经过收尾工具的精度筛选：

```bash
./cpp/build-trt10/trt_lane_benchmark \
  models/rtdetr-l-int8-box-add6-fp32.engine \
  ultralytics/coco/images/val2017/000000147740.jpg \
  results/lane_benchmark_int8 \
  --frames 500 --warmup 30 --rounds 5 --lanes both --poll-us 50
```

2026-10-19最新复测在开测前GPU利用率1%时得到单lane190.43 FPS、双lane189.85 FPS，见
[int8_speed_latest.json](../results/detr_closeout_20261019/int8_speed_latest.json)。
上一轮与历史速度保存在 [int8_speed.json](../results/detr_closeout_20261019/int8_speed.json)，
各次测试外GPU利用率分别记录。

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

默认模型：`models/rtdetr-l-fp32mix16-trt10.engine`（混合 FP16、FP32 I/O）。
默认图片：`ultralytics/coco/images/val2017/000000147740.jpg`。
默认结果：`results/rtdetr_result.jpg`，通过 imwrite 保存。

指定模型、图片和保存位置（这里显式选择历史 INT8 候选）：

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

## 历史 INT8 的 Python 数值对齐

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
