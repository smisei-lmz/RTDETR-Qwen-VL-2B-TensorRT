# INT8 掉点位置诊断

## 已定位的位置

主要敏感组：Decoder 的六轮框坐标更新 Add：

```text
/model.28/decoder/Add
/model.28/decoder/Add_1
/model.28/decoder/Add_2
/model.28/decoder/Add_3
/model.28/decoder/Add_4
/model.28/decoder/Add_5
```

对应 `ultralytics/ultralytics/nn/modules/transformer.py` 中：

```python
refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))
```

六处输入的 inverse-sigmoid 坐标量化步长约为 0.077～0.080。
以坐标 logit 接近零为例，sigmoid 的导数约 0.25：步长 0.08 对应归一化坐标约
0.02 的间隔，即 640 输入上的约 12.8 像素间隔、约 6.4 像素的半步舍入误差。
这是局部量级估算，不表示每个框都有这么大的实际误差。

## 对照方法

- 来源：`models/rtdetr-l-int8.onnx`；SHA256 在各实验目录的 manifest.json。
- 从 val2017 的 4500 张图片中，用 seed=42 固定抽取 200 张。每个模型使用同一列表。
- 使用 ONNX Runtime CUDA、Ultralytics RTDETR 验证，640×640、batch=1、workers=0。
- 将选中消费者的 DQ 输入接回对应 Q 的浮点输入；不改变其他消费者的连接。
- 不重新校准，保留其他节点的原有尺度，清理不再使用的 Q/DQ 和初始化值。
- 这是固定尺度下的量化敏感度实验，不等同于重新校准每个候选配置。
- 不同组之间可能存在误差相互影响，收益不能相加。
- 子集指标只在同一个子集内比较，不与完整验证指标混用。
- 未修改用户的 coco_val.yaml。实验另写配置，明确使用 val2017；用户当前配置指向 calibration。

## 第一轮：算子类型（固定 200 张，mAP50-95）

| 对照 | mAP50-95 |
| --- | ---: |
| FP32 ONNX | 58.05% |
| 原始扩大范围 INT8 ONNX | 43.24% |
| 仅恢复 MatMul 输入为浮点 | 44.28% |
| 仅恢复 Gemm 输入为浮点 | 43.27% |
| 仅恢复 Add 输入为浮点 | 54.74% |
| 仅恢复 Mul 输入为浮点 | 44.84% |
| 恢复 Clip（实际没有输入 DQ 可撤销，零改动对照） | 43.24% |
| 仅保留 Conv/MatMul/Gemm 消费者的原有 Q/DQ 输入 | 57.19% |

第一轮指向 Add 组。完整指标在 `runs/quant_sensitivity/summary.json`。

## 第二轮：Add 的位置（同一 200 张）

| 仅恢复的 Add 位置 | 修改节点数 | mAP50-95 |
| --- | ---: | ---: |
| Decoder 六轮框坐标更新 | 6 | 52.69% |
| Deformable Attention 采样位置相加 | 6 | 44.17% |
| Decoder bbox head 内部偏置相加 | 18 | 43.85% |
| Encoder 检测头相关 Add | 6 | 45.39% |
| Decoder 层和 query position head 的 Add | 102 | 43.70% |
| Backbone/neck 的 Add | 15 | 43.14% |

完整指标和修改节点名单在 `runs/quant_sensitivity_add/`。

## 第三轮：六轮框更新逐个恢复（同一 200 张）

| 唯一恢复的节点 | mAP50-95 |
| --- | ---: |
| Add | 45.36% |
| Add_1 | 46.04% |
| Add_2 | 46.33% |
| Add_3 | 44.87% |
| Add_4 | 44.20% |
| Add_5 | 45.05% |

单个节点恢复不及六处一起恢复，结果支持多轮坐标量化误差累积。
六处是主要敏感组，尚不能解释全部剩余精度差距。
完整指标在 `runs/quant_sensitivity_box_single/summary.json`。

## 校准数据检查

`calibration.npy` 为 `[100,3,640,640]`、float32、范围 [0,1]，无 NaN/Inf。
前三张与当前直接 resize→RGB→CHW→除以255 的制作结果逐元素一致。
图中初始化值提供的 Q/DQ scale 均为正数且有限。
结果在 `runs/quant_sensitivity/calibration_checks.json`。
这些检查不证明样本分布充分；本轮没有测试扩大校准数据或更换校准算法。

## 完整 4500 张验证

仅恢复上述六个 Decoder 框更新 Add 的输入为浮点，其余保持原量化图：

| 模型/后端 | mAP50-95 | mAP50 |
| --- | ---: | ---: |
| 六处保护的 ONNX，ORT CUDA，本次完整验证 | 46.94% | 67.96% |
| 原扩大范围 INT8 engine，TRT 10，用户此前完整日志 | 38.13% | 62.38% |
| FP32 engine，TRT 11，此前完整基线 | 52.21% | 70.61% |

此表包含不同后端的历史结果，作为背景参考；本次严格同后端因果对照使用上面的
200 张 ONNX 实验。全量复核确认保护六处坐标运算可明显恢复精度，但尚未恢复到 FP32。
原量化 ONNX 约 38% 是用户报告，本轮没有再完整重复这一基线。

完整指标、日志、4500 张图清单及配置在 `runs/quant_sensitivity_full/`。
可使用以下命令单独复验已保存的候选模型：

```bash
.venv-trt10/bin/python detr_val.py --model models/diagnostics/rtdetr-l-int8-box-add6-fp32.onnx --data runs/quant_sensitivity_full/val_4500_seed42.yaml --batch 1 --device 0 --workers 0 --name val_box_add6_full
```

## 复现

```bash
.venv-trt10/bin/python detr_quant_sensitivity.py
.venv-trt10/bin/python detr_quant_sensitivity.py --cases runs/quant_sensitivity/add_cases.json --directory runs/quant_sensitivity_add
.venv-trt10/bin/python detr_quant_sensitivity.py --cases runs/quant_sensitivity/add_single_cases.json --directory runs/quant_sensitivity_box_single
.venv-trt10/bin/python detr_quant_sensitivity.py --count 4500 --cases runs/quant_sensitivity/full_cases.json --directory runs/quant_sensitivity_full --keep-models
```

候选对照恢复六个框更新 Add 的浮点输入，其余原有 Q/DQ 保留。
它用于验证掉点位置，包含浮点坐标运算。

独立副本：`models/diagnostics/rtdetr-l-int8-box-add6-fp32.onnx`。
该模型仍保留 Conv 110/110、MatMul 96/111、Gemm 7/7 的两个计算输入上的 Q/DQ。
这里统计的是图覆盖数量，不代表引擎实际内核精度。

## 量化脚本中的保护选项

已有量化 ONNX 可直接处理，不需要重新校准：

```bash
.venv-trt10/bin/python detr_int8_quantize.py --restore-box-add-from models/rtdetr-l-int8.onnx --output models/rtdetr-l-int8-box-add6-fp32.onnx
```

该命令输出已经与上述完整验证模型做过 SHA256 比较，字节一致。

从 FP32 重新校准时：

```bash
.venv-trt10/bin/python detr_int8_quantize.py --onnx models/rtdetr-l-fp32.onnx --protect-box-add --output models/rtdetr-l-int8-box-add6-fp32.onnx
```

实现使用 ModelOpt `nodes_to_exclude=[r"^/model\.28/decoder/Add(?:_[1-5])?$"]`，
并在保存后检查、绕过这六处仍可能与其他消费者共享的输入 Q/DQ。
重新校准路径没有在本轮完整重跑，精度应以新生成模型的 val 为准。
