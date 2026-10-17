# RT-DETR INT8 验证记录

## 已验证的运行方案

使用项目内 `.venv-trt10/bin/python`，TensorRT 版本为 **10.13.3.9**。
该环境继承现有 conda torch 环境的 PyTorch 等依赖，并单独安装 TensorRT 10。

量化文件：`weights/rtdetr-l-int8-fp32.onnx`。
可用 engine：`weights/rtdetr-l-int8-fp32-trt10.engine`。
量化配置是 **Conv INT8，其他运算保留 FP32**，输入输出均为 FP32。

## 完整验证结果

数据：`coco_val.yaml` 指向的本地 4500 张 COCO val2017 图片；80 类；
输入 640×640；batch=1；device=0；workers=0；使用默认验证置信度。
GPU：NVIDIA GeForce RTX 5060。PyTorch：2.14.0+cu132。

| 模型 | TensorRT | mAP50-95 | mAP50 | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| FP32 engine 基线 | 11.3.0.99 | 0.5221 | 0.7061 | 0.7392 | 0.6559 |
| Conv INT8 + FP32 engine | 10.13.3.9 | 0.5178 | 0.7020 | 0.7434 | 0.6467 |

mAP50-95 相差 0.0043，即约 0.43 个百分点。
完整日志分别保存在 `runs/detect/val_fp32_baseline/validation.log` 和
`runs/detect/val_int8_trt10/validation.log`。

## 诊断证据与边界

- 新的 calibration.npy 已确认使用直接 resize、RGB、CHW、float32、除以 255；
  形状为 [100,3,640,640]，数据范围 [0,1]。
- 同一份 Conv-only INT8 + FP32 ONNX 在 TensorRT 11.3 和 10.13.3 中构建。
  11.3 的单图原始输出明显偏离 ONNX，10.13.3 恢复正确检测。
- 另一个保留 stem2a 为 FP32 的诊断模型，在相同 100 张图上，
  ONNX Runtime CUDA 的 mAP50-95 为 0.5730，TensorRT 11.3 为 0.0405；
  同一子集 FP32 engine 基线为 0.5820。这些子集数字不与上面的完整验证数字混用。
- 保留所有未量化计算为 FP32、调整构建优化级别、保护部分特殊卷积，
  都没有单独解决 TensorRT 11.3 的异常。
- 结果支持“当前 GPU/软件组合下存在版本相关的 TensorRT 量化执行异常”。
  尚未定位具体内部内核，也不据此推断所有 TensorRT 11 环境都有同一问题。
- `weights/diagnostics/` 中是失败的对照实验文件，不是已验证的部署模型。

## 复现命令

在 RTDETR-ONNX-TensorRT 项目目录运行：

```bash
# 已有模型直接验证
.venv-trt10/bin/python detr_val.py --model weights/rtdetr-l-int8-fp32-trt10.engine --batch 1 --device 0 --workers 0 --name val_int8_trt10

# 重新制作校准数据、量化并构建
.venv-trt10/bin/python prepare_calibration.py
.venv-trt10/bin/python detr_int8_quantize.py --op-types Conv --output weights/rtdetr-l-int8-fp32.onnx
.venv-trt10/bin/python detr_export_engine.py --onnx weights/rtdetr-l-int8-fp32.onnx --output weights/rtdetr-l-int8-fp32-trt10.engine
```

该 engine 应使用上述 TensorRT 10 环境加载。

## 扩大到所有支持的算子类型

`detr_int8_quantize.py` 默认申请量化安装的 ModelOpt 所支持的全部类型，
包括 Conv、MatMul、Gemm、Add、Mul、池化等。ModelOpt 仍可能排除特定节点；
Softmax 等不支持的运算保留浮点，因此不能称为所有运算均为 INT8。
量化结束会打印各类型接入 DQ 的节点数。这是 ONNX 图覆盖情况，不等于引擎实际内核精度。

```bash
.venv-trt10/bin/python detr_int8_quantize.py
.venv-trt10/bin/python detr_export_engine.py --onnx weights/rtdetr-l-int8-all.onnx --output weights/rtdetr-l-int8-all-trt10.engine
.venv-trt10/bin/python detr_val.py --model weights/rtdetr-l-int8-all-trt10.engine --batch 1 --device 0 --workers 0 --name val_int8_all_trt10
```

上述扩大范围的配置后续已构建和验证，并出现明显掉点；不沿用 Conv-only 的精度结果。
固定尺度的逐组、逐位置敏感度对照见 [QUANT_SENSITIVITY.md](QUANT_SENSITIVITY.md)。
