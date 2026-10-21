# Day 23 · 先打通 VLM 通信闭环

> 旧版实验入口，编号不对应新版时间线。请先按 [6个大任务计划](../../每日开发路线.md) 选择需要复用的实验；本目录run.py仍执行原有流程。


阶段：ROS2 与 VLM。先修：22。

## 今日学习与实现

需要掌握：Action goal/feedback/result、single-flight。

使用明确标注的 mock VLM，自动发送问题，接收反馈和回答；追踪 request_id 与快照身份。

本目录的 `run.py` 是当天可执行实验入口；具体实现已写在下面列出的共享源文件中。先运行，再逐段阅读，最后完成上面的参数对比/修改任务。每日共享实现是为了让你维护同一套完整系统，而不是复制 37 份不一致的代码。

## 运行

在项目根目录、已激活的 WSL Python 环境中执行（板端阶段改为板上终端）：

```bash
python days/day23/run.py
```



正常运行结束应返回 0；故障实验的预期错误由课程入口捕获并核对。

运行参数会保存到 `runs/day23/invocation.json`。实际实验数据写入该目录，模型共享产物位于 `artifacts/`，图片位于 `data/`；具体命令也会打印。自动脚本返回成功仅表示命令成功，下面的人工验收仍需完成。

## 要读和修改的代码

- [ros_ws/src/edge_assistant/edge_assistant/backends.py](../../ros_ws/src/edge_assistant/edge_assistant/backends.py)
- [ros_ws/src/edge_assistant/edge_assistant/vlm_server.py](../../ros_ws/src/edge_assistant/edge_assistant/vlm_server.py)
- [ros_ws/src/edge_assistant/edge_assistant/coordinator.py](../../ros_ws/src/edge_assistant/edge_assistant/coordinator.py)
- [tools/ros_observe.py](../../tools/ros_observe.py)
- [ros_ws/src/edge_interfaces/action/AnalyzeScene.action](../../ros_ws/src/edge_interfaces/action/AnalyzeScene.action)
- [tools/course.py](../../tools/course.py)

`tools/course.py` 中 `day==23`（板端还有嵌套分支）给出全部实际调用，不会缺依赖后悄悄改用模拟模型。

## 验收

完整 Action 往返；回答显示 MOCK；超时/新帧不造成答案贴错；本日不验收视觉理解能力。

将命令、环境、实际输出、问题原因和你修改的代码写入 `runs/day23/notes.md`。未安装依赖、跳过、只读过源码与通过验收是四种不同状态。

遇到失败：先保留日志，再检查先修日的产物与[环境搭建](../../环境搭建.md)。平台限制见[架构与验收](../../架构与验收.md)。
