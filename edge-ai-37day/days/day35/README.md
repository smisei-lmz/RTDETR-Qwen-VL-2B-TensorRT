# Day 35 · 板端 TensorRT 完整检测

> 旧版实验入口，编号不对应新版时间线。请先按 [6个大任务计划](../../每日开发路线.md) 选择需要复用的实验；本目录run.py仍执行原有流程。


阶段：Jetson 迁移。先修：34；Jetson 对应 PyTorch/TRT。

## 今日学习与实现

需要掌握：引擎重建、模型对齐、统一内存。

在板上重建引擎和 PyTorch基线；先比较 Python TRT 与参考，再运行共享 camera→CUDA→TRT。

本目录的 `run.py` 是当天可执行实验入口；具体实现已写在下面列出的共享源文件中。先运行，再逐段阅读，最后完成上面的参数对比/修改任务。每日共享实现是为了让你维护同一套完整系统，而不是复制 37 份不一致的代码。

## 运行

在项目根目录、已激活的 WSL Python 环境中执行（板端阶段改为板上终端）：

```bash
python days/day35/run.py
```



正常运行结束应返回 0；故障实验的预期错误由课程入口捕获并核对。 必须在 Jetson 上运行；PC 不会伪装板端结果。

运行参数会保存到 `runs/day35/invocation.json`。实际实验数据写入该目录，模型共享产物位于 `artifacts/`，图片位于 `data/`；具体命令也会打印。自动脚本返回成功仅表示命令成功，下面的人工验收仍需完成。

## 要读和修改的代码

- [jetson/v4l2_camera.hpp](../../jetson/v4l2_camera.hpp)
- [jetson/camera_run.cpp](../../jetson/camera_run.cpp)
- [jetson/nv_pool.hpp](../../jetson/nv_pool.hpp)
- [pc/src/cuda_backend.cu](../../pc/src/cuda_backend.cu)
- [tools/detr_tools.py](../../tools/detr_tools.py)
- [config.example.json](../../config.example.json)
- [tools/course.py](../../tools/course.py)
- [ros_ws/src/edge_assistant/edge_assistant/rtdetr_backend.py](../../ros_ws/src/edge_assistant/edge_assistant/rtdetr_backend.py)
- [RT-DETR部署说明.md](../../RT-DETR部署说明.md)

`tools/course.py` 中 `day==35`（板端还有嵌套分支）给出全部实际调用，不会缺依赖后悄悄改用模拟模型。

## 验收

使用 artifacts/jetson-models 的引擎；真实场景检测；无 source 早归还；记录板端精度与耗时，不复用 PC 性能结论。

将命令、环境、实际输出、问题原因和你修改的代码写入 `runs/day35/notes.md`。未安装依赖、跳过、只读过源码与通过验收是四种不同状态。

遇到失败：先保留日志，再检查先修日的产物与[环境搭建](../../环境搭建.md)。平台限制见[架构与验收](../../架构与验收.md)。
