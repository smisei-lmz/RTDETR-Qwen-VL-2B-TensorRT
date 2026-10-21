# RT-DETR-L 定量分析与简历依据

汇总日期：2026-10-17。平台：PC / RTX 5060 / WSL/Linux。历史 CPU/CUDA 对照来自已保存实验；本次新增 1/2 lane 持续推理及 Nsight Systems 分析。Profiler 数据与正常性能数据分别报告。

2026-10-19 收尾补充见第7节。当前默认引擎已改为通过数据集精度预算的混合 FP16；
前6节的历史 INT8 数值、模型哈希与结论保持原实验范围。

## 已实现的优化和证据

| 实现 | 代码 | 证据与边界 |
| --- | --- | --- |
| 共享 TensorRT engine，独立 context、主 stream、buffer | load_TrtEngine_runtime.h/.cpp、laneManager.h/.cpp | 双 lane 正确性通过；没有量化显存节省 |
| 融合 CUDA resize、BGR→RGB、HWC→CHW、归一化 | src/CudaPreprocess.cu | 接入已有融合实现；没有分步 CUDA 对照，不单独归因融合收益 |
| 上传 uint8 原图，在 GPU 生成输入张量 | lane::submit、benchmark.cpp | 本例 640×480 原图的 H2D 字节量减少 81.25% |
| 锁页暂存、异步传输、容量复用 | lane::submit、BufferManager | 稳态时间线无 CUDA 分配/释放；两种历史对照均使用锁页内存，未隔离其收益 |
| CUDA 筛选、框转换和稳定压缩 | src/CudaPostprocess.cu | 功能通过；历史 300 框负载下 GPU 后处理本身比 CPU 更慢 |
| Event 与 Idle/Running/Ready/Failed 状态管理 | laneManager.cpp | 结果取走后回收；部分提交失败不能直接复用；析构等待在途任务 |
| Q/DQ 算子组、位置和节点消融 | detr_quant_sensitivity.py、detr_int8_quantize.py | 同子集、同后端精度恢复；不等同于当前 engine 全量 mAP |
| 持续测量、逐帧日志、NVTX 与 Nsight Systems | lane_benchmark.cpp、analyze_optimizations.py | 实测配置选择依据，保留无收益的双 lane 结果 |

## 1. 已有 CPU / CUDA 前后处理对照

来源：[实验说明](BENCHMARK.md)、[摘要](../results/benchmark_single_lane.summary.json)、[逐帧 CSV](../results/benchmark_single_lane.samples.csv)。

条件：RTX 5060，TensorRT 10.13.3.9，OpenCV CPU 线程数 1，batch=1，原图 640×480，模型输入 640×640。每种模式预热 30 帧、测量 200 帧，交替运行。包含准备、传输、推理与前后处理；不含读图、模型加载、初始化。

| 指标 | CPU 链路 | CUDA 链路 | 变化 |
| --- | ---: | ---: | ---: |
| 平均链路延迟 | 7.9811 ms | 5.8730 ms | −26.41%，约 1.36× |
| P95 链路延迟 | 10.1107 ms | 7.2246 ms | −28.55% |
| CPU 准备阶段 | 1.9021 ms | 0.1361 ms | CUDA 路径这里只暂存原图 |
| GPU 前处理 | 无 | 0.0822 ms | 前处理移到 GPU |
| H2D 数据量/帧 | 4,915,200 B | 921,600 B | −81.25% |
| H2D 阶段耗时 | 0.3908 ms | 0.0764 ms | −80.45% |
| TensorRT 区间 | 5.5331 ms | 5.4532 ms | 不是主要收益来源 |
| CPU / GPU 后处理 | 0.0030 ms | 0.0092 ms | 此负载 GPU 更慢 |

输入预处理最大差 5.96e-8；两张检查图片的原始输出及检测框对照通过。不是视频端到端延迟，也不是新的全数据集 mAP。

81.25% 依赖原图分辨率：上传 640×480×3 个 uint8，替代上传 640×640×3 个 float，不能外推为任意分辨率的固定降幅。整体收益来自预处理和传输的组合变化，不等于模型推理本身加速 26.4%。

## 2. 本次新增 1/2 lane 对比

来源：[摘要](../results/lane_benchmark_20261017.summary.json)、[5000 个样本](../results/lane_benchmark_20261017.samples.csv)、[环境与哈希](../results/lane_benchmark_20261017.environment.json)。

模型为 models/rtdetr-l-int8-box-add6-fp32.engine，SHA256：
85cae6fa0fb1ae8e8a466da6c5a7038179ac8aefa455a592a6db3cbf24aa7bb2。
与历史前后处理对照的 engine 哈希相同，但两次实验分别报告，不拼接跨轮收益。

测量方法：

- 一个 CPU 调度线程，每条 lane 独立执行资源，共享一个 engine。
- 每种配置 5 轮，每轮 500 帧，每条 lane 每轮分别预热 30 帧。
- 顺序为 1→2、2→1、1→2、2→1、1→2。
- 固定图片从内存重复输入，CUDA 前后处理，阈值 0.25、80 类。
- 无进展时 sleep_for(50 μs)；只调整轮询频率，完成仍由 event 确认。
- 吞吐 = 总完成帧数 / 总墙钟时间；不用 1000/平均单帧延迟计算多 lane FPS。
- 延迟为 submit 开始到 collect 返回结果，不含读图、准入前等待和 ROS。
- 各配置的 2500 个样本合并，P50/P95/P99 使用 nearest-rank。

| 指标 | 1 lane | 2 lane |
| --- | ---: | ---: |
| 完成帧数 | 2500 | 2500 |
| 聚合吞吐 | 180.57 FPS | 179.92 FPS |
| 各轮 FPS 范围 | 167.76–186.12 | 172.36–186.71 |
| 平均服务延迟 | 5.538 ms | 11.097 ms |
| P50 | 5.378 ms | 10.971 ms |
| P95 | 6.605 ms | 12.675 ms |
| P99 | 8.150 ms | 14.784 ms |

结论：2 lane 聚合吞吐差约 −0.36%，远小于轮间波动，不解释为确定的吞吐变化；平均延迟约翻倍。当前固定图负载选择 1 lane，其 P99 相对 2 lane 低约 44.9%。这是配置选择，不是模型计算加速。

5000 个帧身份完整、无重复、无失败。闭环测试只向空闲 lane 提交，没有固定采集节拍和上游队列，零丢弃不证明真实视频无丢帧。未测量或宣称显存峰值下降。

## 3. Nsight Systems 诊断

分别采集预热后 100 帧，用 CUDA Profiler API 限定窗口。完整报告在 results/profiles/，已忽略以免大文件进入 Git。[轻量分析 JSON](../results/optimization_analysis_20261017.json) 保留可核对指标。

| 诊断指标 | 1 lane | 2 lane |
| --- | ---: | ---: |
| GPU kernel 实例/帧 | 378 | 378 |
| Kernel 区间并集 / 录制窗口 | 38.96% | 45.42% |
| 跨 lane kernel 重叠 / 窗口 | 不适用 | 0% |
| Kernel 在对应 CPU launch API 返回后的间隔 P50 | 0.338 ms | 4.931 ms |
| 稳态 CUDA 分配/释放调用数 | 0 | 0 |

lane.trt_enqueue 的 CPU NVTX 范围合计约 549.59 ms、524.41 ms，包含大量 launch API。当前证据支持先研究提交、排队与调度，而不是直接认定自定义 kernel 的访存是主瓶颈。

边界：

- CPU enqueue 范围包含驱动等待和 profiler 开销，不是纯 GPU 计算时间。
- Kernel 时间覆盖率不是 SM 利用率，不能据此声称 GPU 算力满载。
- TensorRT 有内部辅助 stream。脚本通过 CPU submit 范围及 H2D 主 stream 将每份报告的 37800 个 kernel 全部关联到 lane，再计算跨 lane 重叠。
- API 返回后间隔定义为 max(0, GPU kernel start − CPU API end)，不包含 API 内部等待。
- 这次采样没有跨 lane kernel 重叠，并有更长的排队间隔；确切的驱动、launch 或隐式依赖原因仍需单独实验。
- Profiler 改变时序，本节不使用其 FPS 作为性能结论。
- 本次没有做 Nsight Compute 访存分析，也没有实施 CUDA Graph。

## 4. 量化优化的独立证据

来源：[诊断说明](../QUANT_SENSITIVITY.md)、[算子组对照](../runs/quant_sensitivity/summary.json)、[Add 位置对照](../runs/quant_sensitivity_add/summary.json)。

同一 200 张 COCO 子集，同一 ORT CUDA 后端，固定其他 Q/DQ 尺度：

| 配置 | mAP50–95 |
| --- | ---: |
| 原扩大范围 INT8 ONNX | 43.2441% |
| 保护 6 处 decoder 框更新 Add 的浮点输入 | 52.6874% |
| 精度恢复 | +9.4433 个百分点 |

方法是按算子类型分组、按 Add 位置分组，再做六个敏感节点的逐个消融。这是子集敏感度对照，不是完整验证集或当前 TensorRT engine 的 mAP。已有 4500 张记录涉及不同模型和后端，不把它们直接拼成严格因果对比。

## 5. 简历表述

建议项目名：**RT-DETR-L C++ / TensorRT 推理部署与 CUDA 运行时优化**。

- 搭建 C++/TensorRT 检测后端，集成融合 CUDA 预处理及 uint8 原图异步上传；RTX 5060、batch=1 固定图热态测试中，链路平均延迟由 7.98 ms 降至 5.87 ms（−26.4%），P95 降低 28.5%。
- 实现共享 engine、独立 context/stream/buffer 的多 lane 运行时，使用 CUDA Event 管理提交、完成查询与缓冲回收；完成 5000 帧性能对照及 Nsight Systems 分析，识别双 lane 增加排队而未提升吞吐，选择单 lane 作为当前负载配置。
- 通过 Q/DQ 算子组、位置和节点消融，定位 6 处 decoder 框更新 Add 的量化敏感性；同一 200 张 COCO 子集、ORT CUDA 后端，mAP50–95 从 43.24% 恢复至 52.69%（约 +9.4 个百分点）。

面试准备：解释输入字节量为何减少、pinned memory 收益为何未被隔离、stream/event 与内存生命周期、GPU 后处理为何可能更慢、多 stream 为何不等于吞吐翻倍、子集与全量及 ORT/TRT 的边界。

ROS、VLM、Jetson、真实视频丢帧和双模型调度属于后续工作。

## 6. 复现

```bash
cmake --build cpp/build-trt10 -j 4
./cpp/build-trt10/trt_lane_check
./cpp/build-trt10/trt_lane_benchmark \
  models/rtdetr-l-int8-box-add6-fp32.engine \
  ultralytics/coco/images/val2017/000000147740.jpg \
  results/lane_benchmark_20261017 \
  --frames 500 --warmup 30 --rounds 5 --lanes both --poll-us 50
```

时间线分别对 lane 1、2 运行；下面展示 lane 2，另一份对应替换 lane 数和输出名：

```bash
mkdir -p results/profiles
nsys profile \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o results/profiles/lane2_20261017 \
  ./cpp/build-trt10/trt_lane_benchmark \
  models/rtdetr-l-int8-box-add6-fp32.engine \
  ultralytics/coco/images/val2017/000000147740.jpg \
  results/profiles/lane2_20261017_run \
  --frames 100 --warmup 30 --rounds 1 --lanes 2 --poll-us 50 --profile 1

nsys export --type sqlite \
  --output results/profiles/lane2_20261017.sqlite \
  results/profiles/lane2_20261017.nsys-rep
```

两个 SQLite 导出文件齐备后，重算指标和图表：

```bash
.venv-trt10/bin/python cpp/analyze_optimizations.py
```

[性能图表](../results/optimization_analysis_20261017.png)。

正常性能和 profiler 分开运行。再次生成 profiler 报告时更换输出名或明确处理已有文件。分析脚本固定本次数据路径、源图和模型哈希检查，用于复现本报告，不是通用分析器。

## 7. DETR 收尾与正式配置

日期：2026-10-19。汇总见 [closeout.json](../results/detr_closeout_20261019/closeout.json)，
图表见 [overview.png](../results/detr_closeout_20261019/overview.png)。
固定 Ultralytics RT-DETR-L、batch=1、640×640 输入、FP32 I/O、COCO80，
部署配置见 [deployment.lock.json](../results/detr_closeout_20261019/deployment.lock.json)。

旧 FP32/FP16 engine 无法由当前 TensorRT 10.13.3.9 反序列化，已从本地 ONNX
重建带 `-trt10` 后缀的文件，保留旧文件。构建脚本现在保存源图/引擎哈希、
TensorRT 版本、强类型构建和详细检查信息；[引擎清单](../results/detr_closeout_20261019/engines.json)
记录实际 I/O 和张量格式。混合 FP16 有 Half 张量，I/O 仍为 FP32；
张量存储类型不等于逐条指令的运算精度，FP32 参考未额外关闭 TensorRT 默认 TF32 策略。

### 同条件数据集精度

三种冻结候选均使用实际 C++ lane：CUDA 前处理 → TensorRT 10 → CUDA 解码。
使用本地全部4500张带标注的 COCO val2017 图片，图片/标注哈希保存在
[数据清单](../results/detr_closeout_20261019/dataset.json)。这是5000张官方验证集中的本地可用部分。
采用 pycocotools 2.0.11 bbox COCOeval，阈值0.001、maxDets=100，导出框裁至图像边界，不加NMS。
零检测图片仍参加评估，不从分母排除。

| 候选 | mAP50–95 | AP50 | 相对 FP32 下降 |
| --- | ---: | ---: | ---: |
| FP32 图 / TRT10 | 52.7038% | 71.1587% | — |
| 混合 FP16 / TRT10 | 52.6747% | 71.1834% | 0.0291 个百分点 |
| 保护六处 Add 的 INT8 / TRT10 | 47.6880% | 68.4927% | 5.0158 个百分点 |

预算在本轮评估前固定为最多下降1个百分点。选择混合 FP16 作为后续默认引擎，
INT8保留作实验候选。当前图来自已有本地模型产物，本轮是候选部署质量对照；
严格固定尺度的量化因果消融仍以第4节的同后端200张实验为依据。
本轮官方 COCOeval 与历史 Ultralytics/ORT 统计分别报告，不拼接为跨实验收益。

### 新默认引擎的性能

环境与历史实验同为 RTX 5060、TensorRT 10.13.3.9、OpenCV 4.5.4、CPU线程1。
固定640×480图片，预热后交替测200次/模式，范围包含前后处理、传输、推理，
不含读图与初始化。详情见 [前后处理摘要](../results/detr_closeout_20261019/preprocessing.json)。

| 指标 | CPU 前后处理链路 | CUDA 前后处理链路 |
| --- | ---: | ---: |
| 平均延迟 | 11.4110 ms | 9.3194 ms |
| P95 | 13.3757 ms | 10.9138 ms |
| H2D/帧 | 4,915,200 B | 921,600 B |

链路平均延迟降低18.33%、P95降低18.41%；这是前后处理与传输的组合收益。
预处理最大绝对误差5.96e-8，原始输出差与解码框差均为0。

五轮交替1/2 lane、每配置每轮500帧，见
[lane摘要](../results/detr_closeout_20261019/lane_benchmark.json)。

| 指标 | 1 lane | 2 lane |
| --- | ---: | ---: |
| 完成帧 | 2500 | 2500 |
| 聚合吞吐 | 110.69 FPS | 112.81 FPS |
| 平均 submit→collect 延迟 | 9.0341 ms | 17.7040 ms |
| P95 | 10.2502 ms | 19.8562 ms |
| P99 | 11.2937 ms | 21.8786 ms |

双 lane 吞吐仅高约1.92%，尾延迟明显增加，默认采用1 lane。
5000条逐帧身份已核对，无重复/缺失；闭环测试没有上游到达队列，不代表实际视频零丢帧。
这组FP16结果与第2节的INT8结果分别报告。

### 运行与退出检查

默认FP16与INT8候选的实际GPU lane检查通过：0.001评估阈值、CPU参考、输入覆盖、
非连续ROI、忙/就绪状态、复用、独立输出与在途任务析构，见
[正确性记录](../results/detr_closeout_20261019/correctness.json)。

30分钟长测按用户要求中止，保留20985帧完整日志和截至190.08秒的内存采样，
见 [中断记录](../results/detr_closeout_20261019/stability_interrupted.json)。
采样窗口进程RSS范围约116 KiB，后半段RSS趋势为0 KiB/min，设备可用显存采样无变化。
这些是短窗口观察，不作为长期稳定性通过证明；信号中断也不作为优雅退出证明。

另执行5.01秒短检查，完成550帧，每帧结果一致性检查通过，所有结果取走后正常销毁lane。
销毁后全设备可用显存回升约74 MiB；engine仍驻留，CUDA模块/缓存可以保留。
RSS属于C++进程，显存来自cudaMemGetInfo的全设备视角，不宣称进程显存峰值。
本次收尾明确豁免30分钟要求；[短检查记录](../results/detr_closeout_20261019/stability.json)
保留 `plan_30_minute_requirement_met=false`。

### 下一阶段交接

`trt_example`、前后处理benchmark、lane benchmark和lane检查默认选择新FP16引擎。
历史性能摘要不会被新默认输出路径覆盖。`LaneConfig` 支持置信阈值与类别数，
保留 `submit/query/collect`；调用方单线程拥有lane，或自行串行化访问。
SharedTrtEngine必须比lane活得更久，collect成功返回的检测数据独立于lane缓冲。

工位视频标注/事件验收、ROS封装、VLM共运行和Jetson仍属后续阶段。
完整复现入口见 [README](README.md)，大体积原始文件在已忽略的 `results/detr_closeout_20261019/raw/`。

## 8. INT8 速度独立补测

2026-10-19 按用户要求独立测量 INT8 速度，不以精度预算筛选或阻止测速。
使用 `rtdetr-l-int8-box-add6-fp32.engine`，模型哈希与第2节一致；
与本轮FP16使用同一二进制、同一640×480图片、OpenCV线程1、每lane预热30帧、
每配置每轮500帧、五轮交替1/2 lane、空闲轮询间隔50μs。
两种配置各完成2500帧，5000条帧身份完整且无重复。

| 指标 | 本轮 1 lane | 本轮 2 lane |
| --- | ---: | ---: |
| 聚合吞吐 | 84.41 FPS | 88.46 FPS |
| 平均 submit→collect 延迟 | 11.8459 ms | 22.5788 ms |
| P95 | 15.3582 ms | 25.7140 ms |
| P99 | 15.8138 ms | 26.4353 ms |

本轮CPU前后处理链路均值13.3122 ms，CUDA链路均值11.7355 ms；
CUDA链路内TensorRT事件区间均值11.2135 ms。前处理最大差5.96e-8，原始输出及框差为0。
范围不包含读图、视频解码、ROS或VLM，也未执行长时间稳定性测试。

开测前GPU利用率46%；测试进程结束后的三次采样为47%、41%、9%，
GPU状态与历史测量时段不能视为相同。具体下降原因未定位，
不以本轮与早先FP16的独立时段结果计算精度模式加速比。
历史INT8记录仍为单lane180.57 FPS、平均5.5377 ms、P99 8.1504 ms；
双lane179.92 FPS、平均11.0966 ms、P99 14.7845 ms。两次记录均保留，不覆盖或挑选其中一组。

本轮完整摘要、历史参考和采样条件见 [int8_speed.json](../results/detr_closeout_20261019/int8_speed.json)，
模型/二进制哈希及精确命令见 [测速环境](../results/detr_closeout_20261019/int8_speed.environment.json)。

### 同日再次复测

再次使用完全相同的engine、二进制、图片和五轮参数；开测前连续三次GPU利用率均为1%。
本次各配置仍测2500帧，5000条帧身份与P99统计已核对。

| 指标 | 1 lane | 2 lane |
| --- | ---: | ---: |
| 聚合吞吐 | 190.43 FPS | 189.85 FPS |
| 平均 submit→collect 延迟 | 5.2509 ms | 10.5167 ms |
| P95 | 5.9598 ms | 11.5243 ms |
| P99 | 6.9947 ms | 12.0967 ms |

测试结束后三次GPU利用率为11%、2%、1%。本次速度恢复到接近历史基准的范围，
与上一轮开测前46%利用率的环境状态存在明显差异；具体外部负载来源仍未确认。
双lane吞吐没有提高，延迟增加，INT8速度配置也优先单lane。
最新汇总及前一轮对照见 [int8_speed_latest.json](../results/detr_closeout_20261019/int8_speed_latest.json)，
前一轮84.41/88.46 FPS的原始记录继续保留。
