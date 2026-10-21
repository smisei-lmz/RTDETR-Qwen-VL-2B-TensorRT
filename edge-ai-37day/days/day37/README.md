# Day 37 · 板端验收与项目交付

> 旧版实验入口，编号不对应新版时间线。请先按 [6个大任务计划](../../每日开发路线.md) 选择需要复用的实验；本目录run.py仍执行原有流程。


阶段：Jetson 迁移。先修：36。

## 今日学习与实现

需要掌握：稳定性、故障注入、实测对比。

运行30分钟共享路径；手工测试断开相机/恢复启动/退出；另录 ROS+VLM 并发监控；整理基线与共享路径对比。

本目录的 `run.py` 是当天可执行实验入口；具体实现已写在下面列出的共享源文件中。先运行，再逐段阅读，最后完成上面的参数对比/修改任务。每日共享实现是为了让你维护同一套完整系统，而不是复制 37 份不一致的代码。

## 运行

在项目根目录、已激活的 WSL Python 环境中执行（板端阶段改为板上终端）：

```bash
python days/day37/run.py
```

默认30分钟；可用 `--seconds 60` 冒烟。本日自动脚本测相机/TRT，ROS+VLM 并发稳定性需另运行 Day36 并监控。

正常运行结束应返回 0；故障实验的预期错误由课程入口捕获并核对。 必须在 Jetson 上运行；PC 不会伪装板端结果。

运行参数会保存到 `runs/day37/invocation.json`。实际实验数据写入该目录，模型共享产物位于 `artifacts/`，图片位于 `data/`；具体命令也会打印。自动脚本返回成功仅表示命令成功，下面的人工验收仍需完成。

## 要读和修改的代码

- [tools/course.py](../../tools/course.py)
- [tools/analyze.py](../../tools/analyze.py)
- [jetson/v4l2_camera.hpp](../../jetson/v4l2_camera.hpp)
- [jetson/camera_run.cpp](../../jetson/camera_run.cpp)
- [jetson/nv_pool.hpp](../../jetson/nv_pool.hpp)
- [pc/src/cuda_backend.cu](../../pc/src/cuda_backend.cu)

`tools/course.py` 中 `day==37`（板端还有嵌套分支）给出全部实际调用，不会缺依赖后悄悄改用模拟模型。

## 验收

30分钟无越界/早归还/死锁；dequeued=completed+bad；退出 held_sources=0；报告真实性能、失败项和已验证版本。

将命令、环境、实际输出、问题原因和你修改的代码写入 `runs/day37/notes.md`。未安装依赖、跳过、只读过源码与通过验收是四种不同状态。

遇到失败：先保留日志，再检查先修日的产物与[环境搭建](../../环境搭建.md)。平台限制见[架构与验收](../../架构与验收.md)。
