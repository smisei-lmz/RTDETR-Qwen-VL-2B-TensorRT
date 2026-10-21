# Day 04 · 固定模型输入输出契约

> 旧版实验入口，编号不对应新版时间线。请先按 [6个大任务计划](../../每日开发路线.md) 选择需要复用的实验；本目录run.py仍执行原有流程。


阶段：模型与环境。先修：03。

## 今日学习与实现

需要掌握：ONNX、NCHW、静态形状、算子图。

导出静态 batch1、640×640 的 RT-DETR ONNX，图内完成 sigmoid 与 top-k；运行 checker 和 ONNX Runtime 数值对齐。

本目录的 `run.py` 是当天可执行实验入口；具体实现已写在下面列出的共享源文件中。先运行，再逐段阅读，最后完成上面的参数对比/修改任务。每日共享实现是为了让你维护同一套完整系统，而不是复制 37 份不一致的代码。

## 运行

在项目根目录、已激活的 WSL Python 环境中执行（板端阶段改为板上终端）：

```bash
python days/day04/run.py
```



正常运行结束应返回 0；故障实验的预期错误由课程入口捕获并核对。

运行参数会保存到 `runs/day04/invocation.json`。实际实验数据写入该目录，模型共享产物位于 `artifacts/`，图片位于 `data/`；具体命令也会打印。自动脚本返回成功仅表示命令成功，下面的人工验收仍需完成。

## 要读和修改的代码

- [tools/detr_tools.py](../../tools/detr_tools.py)
- [config.example.json](../../config.example.json)
- [tools/course.py](../../tools/course.py)
- [ros_ws/src/edge_assistant/edge_assistant/rtdetr_backend.py](../../ros_ws/src/edge_assistant/edge_assistant/rtdetr_backend.py)
- [RT-DETR部署说明.md](../../RT-DETR部署说明.md)

`tools/course.py` 中 `day==4`（板端还有嵌套分支）给出全部实际调用，不会缺依赖后悄悄改用模拟模型。

契约详见 [RT-DETR 部署说明](../../RT-DETR部署说明.md)。ONNX checker 不证明数值一致，脚本还实际执行 ORT；本机没有运行模型的验证记录见验证说明。

## 验收

detector.onnx 通过 checker 和 ORT 逐框校验；输入 images [1,3,640,640]；输出 rtdetr_detections [1,300,6]，含归一化 xyxy、score、class_id。

将命令、环境、实际输出、问题原因和你修改的代码写入 `runs/day04/notes.md`。未安装依赖、跳过、只读过源码与通过验收是四种不同状态。

遇到失败：先保留日志，再检查先修日的产物与[环境搭建](../../环境搭建.md)。平台限制见[架构与验收](../../架构与验收.md)。
