# 单 lane：OpenCV/CPU 与 CUDA 前后处理对比

## 条件与范围

- GPU：RTX 5060，驱动 610.62；CUDA Toolkit 13.4；TensorRT 10.13.3.9。
- Engine：`models/rtdetr-l-int8-box-add6-fp32.engine`，输入 `[1,3,640,640]`，输出 `[1,300,6]`。
- 图片：`000000147740.jpg`，原图 640×480；阈值 0.25。
- OpenCV 4.5.4，显式设置 CPU 线程数为 1；编译 Release。
- 两条链路串行交替，始终只有一个 lane、一个 stream/context。
- 各预热 30 帧，然后各测 200 帧，交替 CPU先/CUDA先，降低运行顺序偏差。
- 每次都重新执行前处理、传输、推理和后处理，不复用前处理结果。
- 固定图片的热态延迟测试；不是实际视频或双 lane 吞吐结果。

## 链路

CPU/OpenCV：

```text
原图 → resize → cvtColor → convertTo(FP32/255) → split(CHW)
     → H2D → TRT → D2H → CPU分数筛选/框坐标转换 → CPU结果可读
```

CUDA：

```text
原图 → 复制到锁页暂存内存 → H2D → CUDA resize/RGB/CHW/归一化
     → TRT → CUDA分数筛选/框坐标转换/稳定压缩
     → D2H（count和固定容量结果包，一次复制） → CPU结果可读
```

前处理均为直接拉伸、双线性 uint8 resize，然后归一化，不使用 LetterBox。
后处理均无额外 NMS，保留原始 Top-K 顺序及浮点坐标。
CUDA resize 的系数及 uint8 舍入参照
[OpenCV 4.5.4 resize实现](https://github.com/opencv/opencv/blob/4.5.4/modules/imgproc/src/resize.cpp)。
尺寸对应的插值系数只初始化一次，像素 resize 在 GPU 执行。

## 计时与内存

- CPU 阶段和整帧延迟用 steady_clock；CUDA 各段用同一 stream 上的 CUDA Event。
- 最后 D2H 之后记录完成事件，调用 cudaEventSynchronize，确认可读后才读取/复用缓冲区。
- 整帧包括 CPU 暂存、H2D、CUDA运算、TRT、D2H、等待完成、CPU结果整理。
- 不包括读图、模型加载、显存/锁页内存分配、初始化插值表、绘图、保存、统计结果写盘。
- 两条链路使用锁页内存，CPU路径前处理直接写入预分配的锁页输入。
- CPU路径 H2D 为 4,915,200 B，CUDA路径为 921,600 B。
- D2H 分别为 7,200 B 和 7,204 B；CUDA固定容量下载不声称节省输出传输。
- 分段时间的和与总时间不必相等：总时间还包括 CPU 提交和完成事件等待等开销。

## 主要结果（平均 ms）

| 阶段 | OpenCV/CPU链路 | CUDA链路 |
| --- | ---: | ---: |
| CPU准备（CPU路径为完整OpenCV前处理；CUDA路径为原图暂存） | 1.9021 | 0.1361 |
| H2D | 0.3908 | 0.0764 |
| CUDA前处理 | 0.0000 | 0.0822 |
| TensorRT | 5.5331 | 5.4532 |
| CUDA后处理 | 0.0000 | 0.0092 |
| D2H | 0.0650 | 0.0531 |
| CPU解码/结果整理 | 0.0030 | 0.0006 |
| 端到端 | 7.9811 | 5.8730 |

| 整帧统计 | OpenCV/CPU | CUDA |
| --- | ---: | ---: |
| median | 7.6784 ms | 5.6560 ms |
| p95 | 10.1107 ms | 7.2246 ms |

本轮 CUDA 链路平均延迟减少 **26.4%**，约 **1.36×**。
主要收益来自前处理和输入传输，不应称为 TensorRT 模型本身加速。
300个候选框的CPU解码约 3.0 μs，
GPU解码约 9.2 μs，另有CPU整理开销；本例单独后处理CPU更快。
各段均为这次实测，不代表所有图片尺寸、线程数和负载下的固定耗时。

## 数值检查

- 640×480主测试图：输入最大绝对差 5.96046448e-08（归一化浮点舍入量级）。
- 同图原始输出最大绝对差 0，15个检测框的坐标最大差 0。
- 额外检查原图240×320的 `000000057150.jpg`，两个方向都需要 resize；输入差同为5.96e-8，原始输出和39个解码框完全一致。
- 第二张图片只用于数值检查，未预热且只测一次，不用于性能结论。
- 每次运行比较程序都会先检查输入、原始输出，以及同一原始输出的CPU/GPU解码；不通过则停止计时。
- 没有运行新的全验证集mAP，结论限于上述数值检查及当前图性能。

## CPU锁页内存方案补充检查

另做一次同条件配对实验，CPU先在普通内存完成OpenCV前处理，再复制到锁页内存上传。
该次 CPU准备平均 2.3265 ms，CPU链路端到端 8.1691 ms；
同轮 CUDA链路 5.5614 ms。
此方案没有改善CPU路径，故保留直接写锁页输入作为默认对照。
不同轮次时钟/负载有波动，应比较各轮内部配对结果，不能用跨轮耗时拼接计算收益。

## 复现

在项目根目录：

```bash
cmake --build cpp/build-trt10 -j 4
./cpp/build-trt10/trt_benchmark
```

指定输入、重复次数、OpenCV线程数：

```bash
./cpp/build-trt10/trt_benchmark \
  models/rtdetr-l-int8-box-add6-fp32.engine \
  ultralytics/coco/images/val2017/000000147740.jpg \
  results/benchmark_single_lane \
  --iterations 200 --warmup 30 --opencv-threads 1
```

普通内存计算再暂存的CPU选项为 `--cpu-staging 1`；默认0为直接写锁页内存。

结果：`results/benchmark_single_lane.summary.json`、`.samples.csv`。
补充：`results/benchmark_cpu_staged.*`、`results/benchmark_resize_check.*`。
早期手写CPU循环基线单独保存在 `results/benchmark_cpu_scalar_reference.*`，不作为主比较结论。

实现入口：`cpp/benchmark.cpp`，CUDA算子：`cpp/src/CudaPreprocess.cu` 和 `cpp/src/CudaPostprocess.cu`。
原 `trt_example` 仍保留原来的单图流程；完整CPU/CUDA链路比较使用 `trt_benchmark`。
