# Day 30 · PC 工位巡检演示与交付

> 旧版实验入口，编号不对应新版时间线。请先按 [6个大任务计划](../../每日开发路线.md) 选择需要复用的实验；本目录run.py仍执行原有流程。


阶段：PC 验收。先修：29。

## 今日学习与实现

需要掌握：启动配置、系统演示、迁移边界。

用真实视频演示有人、离席、报告与返回流程；另演示手工问题；保存真实图片、事件 JSON、回答、模型版本和检测延迟对比。

本目录的 `run.py` 是当天可执行实验入口；具体实现已写在下面列出的共享源文件中。先运行，再逐段阅读，最后完成上面的参数对比/修改任务。每日共享实现是为了让你维护同一套完整系统，而不是复制 37 份不一致的代码。

## 运行

在项目根目录、已激活的 WSL Python 环境中执行（板端阶段改为板上终端）：

```bash
python days/day30/run.py
```



这是交互运行，观察完成后 Ctrl+C。停止程序并不自动代表验收通过。

运行参数会保存到 `runs/day30/invocation.json`。实际实验数据写入该目录，模型共享产物位于 `artifacts/`，图片位于 `data/`；具体命令也会打印。自动脚本返回成功仅表示命令成功，下面的人工验收仍需完成。

## 要读和修改的代码

- [ros_ws/src/edge_assistant/edge_assistant/cpp_bridge.py](../../ros_ws/src/edge_assistant/edge_assistant/cpp_bridge.py)
- [ros_ws/src/edge_assistant/launch/course.launch.py](../../ros_ws/src/edge_assistant/launch/course.launch.py)
- [ros_ws/src/edge_assistant/edge_assistant/backends.py](../../ros_ws/src/edge_assistant/edge_assistant/backends.py)
- [ros_ws/src/edge_assistant/edge_assistant/vlm_server.py](../../ros_ws/src/edge_assistant/edge_assistant/vlm_server.py)
- [ros_ws/src/edge_assistant/edge_assistant/coordinator.py](../../ros_ws/src/edge_assistant/edge_assistant/coordinator.py)
- [config.example.json](../../config.example.json)
- [tools/course.py](../../tools/course.py)
- [ros_ws/src/edge_assistant/edge_assistant/workstation.py](../../ros_ws/src/edge_assistant/edge_assistant/workstation.py)
- [ros_ws/src/edge_assistant/launch/workstation.launch.py](../../ros_ws/src/edge_assistant/launch/workstation.launch.py)
- [tools/workstation_demo.py](../../tools/workstation_demo.py)
- [工位巡检场景.md](../../工位巡检场景.md)

`tools/course.py` 中 `day==30`（板端还有嵌套分支）给出全部实际调用，不会缺依赖后悄悄改用模拟模型。

本日使用 Day25 相同的离席事件链路；按 [工位巡检场景](../../工位巡检场景.md) 保存六项验收证据。

## 验收

真实 RT-DETR 与 Qwen 联合运行；报告可追溯到触发帧；覆盖工位场景六项检查；不把规则模拟或纯检测稳定性等同于完整场景验收。

将命令、环境、实际输出、问题原因和你修改的代码写入 `runs/day30/notes.md`。未安装依赖、跳过、只读过源码与通过验收是四种不同状态。

遇到失败：先保留日志，再检查先修日的产物与[环境搭建](../../环境搭建.md)。平台限制见[架构与验收](../../架构与验收.md)。
