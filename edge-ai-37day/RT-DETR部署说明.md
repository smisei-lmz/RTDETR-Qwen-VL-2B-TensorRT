# RT-DETRv2-S 部署契约与模型迁移

> 模型契约继续沿用；Day编号是旧实验索引。新版对应 [实施计划](每日开发路线.md) 的任务1–2，不要求按旧37天顺序执行。


默认权重为 `PekingU/rtdetr_v2_r18vd`，约 20M 参数，使用卷积骨干、混合编码器与 Transformer 解码器。这里通过 Transformers 4.49.0 的 RTDetrV2ForObjectDetection 加载；不依赖 Ultralytics。

来源：[RT-DETR 官方项目](https://github.com/lyuwenyu/RT-DETR)、[权重与配置](https://huggingface.co/PekingU/rtdetr_v2_r18vd)、[Transformers RT-DETRv2](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/rt_detr_v2)。

## 固定契约

| 项目 | 本项目约定 |
|---|---|
| 输入名／形状 | `images`，`[1,3,640,640]`，静态 batch1 |
| 输入预处理 | RGB、双线性拉伸、half-pixel 坐标、无抗锯齿、除以 255、无均值方差归一化 |
| 输出名／形状 | `rtdetr_detections`，`[1,300,6]` |
| 每行 | `[x1,y1,x2,y2,score,class_id]`，坐标相对原图宽高归一化 |
| 后处理 | 图内 sigmoid + 全 query/class 对 top-k 300；CPU 置信度过滤、坐标还原；无 NMS |
| 类别 | COCO80 连续 ID 0..79；名称读取模型配置，并转换 COCO 名称别名 |
| 引擎 | `detector.plan` 是无前缀的原生 TensorRT plan，C++ 校验张量名和形状 |

该输出是本项目导出 wrapper 的部署协议，不是任意 RT-DETR ONNX 的通用输出。其他仓库输出的 `labels/boxes/scores` 多张量模型不能直接替换。旧 YOLO engine 会被拒绝；重新运行 Day03–06。

为与手写 CUDA 精确对齐，Python 基线使用 `torch.interpolate(..., align_corners=False)`。官方 PIL 预处理在缩小时可能采用不同的抗锯齿行为，因此不能照搬官方 AP 成绩；必须在数据集上评估本部署约定。PyTorch、ONNX、TRT、C++ 均使用相同约定。

## 转换和对齐

- Day03：真实图片、预热、PyTorch 检测框和标注图、类别映射及版本记录。
- Day04：opset17 ONNX；checker 后使用 ONNX Runtime CPU 执行真实图片，进行逐框对齐。
- Day05：重新导出并校验 ONNX，再由 TensorRT Python API 构建 FP16-allowed 引擎，保存 SHA256。内部精度可混合，不等于所有层或 I/O 都为 FP16。
- Day06：原生 TensorRT Python runner 对齐 PyTorch，并在指定 COCO 数据集上比较 mAP。
- Day17：C++ CUDA 预处理误差与 Python 检测结果对齐。

图含 attention、GridSample 和 TopK 等算子；选定的 CUDA/TRT/ONNX 版本必须实际构建和运行。解析失败会明确报错，不能静默切换模型或后端。没有宣称此环境已完成模型导出或 GPU 推理。

## COCO 格式评估数据

创建自己的 `data/coco_eval.json`，例如：

```json
{
  "images": "/absolute/path/coco/val2017",
  "annotations": "/absolute/path/coco/annotations/instances_val2017.json"
}
```

路径也可相对该 manifest 所在目录。然后执行：

```bash
python tools/model_tools.py evaluate --dataset data/coco_eval.json
```

不再自动下载 coco128，也不接收 Ultralytics 数据集 YAML。小规模部署回归可使用有 COCO 标注的子集，但不能将它当成完整 COCO AP 或真实工位泛化成绩。工位事件误触发/漏触发需要额外视频测试。

## 工程边界

原采集线程、图像池、有界队列、每 lane 的 CUDA stream 与 TensorRT context、Jetson 相机共享路径继续保留。模型输出仅 300×6 个值，先直接 D2H，再 CPU 过滤和坐标还原；`gpu_post_d2h_ms` 在真实 DETR 模式中记录这次 D2H，`cpu_ms` 包含解码。没有继续执行旧的 GPU YOLO 候选解码或 CPU NMS。

Qwen2-VL 接收独立 JPEG，不共享 DETR tensor。默认 C++ 快照宽度改为最多 640，以改善工位物品可见性；实际显存、快照复制开销和回答质量仍需量测。
