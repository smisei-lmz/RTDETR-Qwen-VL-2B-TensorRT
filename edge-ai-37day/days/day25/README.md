# Day 25 · 工位离席巡检与 VLM 并行运行

> 旧版实验入口，编号不对应新版时间线。请先按 [6个大任务计划](../../每日开发路线.md) 选择需要复用的实验；本目录run.py仍执行原有流程。


阶段：ROS2 与 VLM。先修：24；17；19 的真实工位视频。

## 今日学习与实现

需要掌握：占用/离席状态、事件快照、VLM 按需调用、GPU 资源竞争。

先跑无模型状态演示，再用真实 TRT+ROS+Qwen 回放工位视频；确认有人2秒、持续无人10秒后触发一次巡检，保存图片和报告。

本目录的 `run.py` 是当天可执行实验入口；具体实现已写在下面列出的共享源文件中。先运行，再逐段阅读，最后完成上面的参数对比/修改任务。每日共享实现是为了让你维护同一套完整系统，而不是复制 37 份不一致的代码。

## 运行

在项目根目录、已激活的 WSL Python 环境中执行（板端阶段改为板上终端）：

```bash
python days/day25/run.py
```



这是交互运行，观察完成后 Ctrl+C。停止程序并不自动代表验收通过。

运行参数会保存到 `runs/day25/invocation.json`。实际实验数据写入该目录，模型共享产物位于 `artifacts/`，图片位于 `data/`；具体命令也会打印。自动脚本返回成功仅表示命令成功，下面的人工验收仍需完成。

## 要读和修改的代码

- [ros_ws/src/edge_assistant/edge_assistant/cpp_bridge.py](../../ros_ws/src/edge_assistant/edge_assistant/cpp_bridge.py)
- [ros_ws/src/edge_assistant/launch/course.launch.py](../../ros_ws/src/edge_assistant/launch/course.launch.py)
- [ros_ws/src/edge_assistant/edge_assistant/backends.py](../../ros_ws/src/edge_assistant/edge_assistant/backends.py)
- [ros_ws/src/edge_assistant/edge_assistant/vlm_server.py](../../ros_ws/src/edge_assistant/edge_assistant/vlm_server.py)
- [ros_ws/src/edge_assistant/edge_assistant/coordinator.py](../../ros_ws/src/edge_assistant/edge_assistant/coordinator.py)
- [tools/course.py](../../tools/course.py)
- [ros_ws/src/edge_assistant/edge_assistant/workstation.py](../../ros_ws/src/edge_assistant/edge_assistant/workstation.py)
- [ros_ws/src/edge_assistant/launch/workstation.launch.py](../../ros_ws/src/edge_assistant/launch/workstation.launch.py)
- [tools/workstation_demo.py](../../tools/workstation_demo.py)
- [工位巡检场景.md](../../工位巡检场景.md)

`tools/course.py` 中 `day==25`（板端还有嵌套分支）给出全部实际调用，不会缺依赖后悄悄改用模拟模型。

先运行 `python tools/workstation_demo.py` 理解规则（没有模型推理），再运行本日真实流程。事件保存在 `runs/workstation/`。完整素材准备、ROI 和六项验收见 [工位巡检场景](../../工位巡检场景.md)。

## 验收

空场景和短暂遮挡不触发；每次离席仅一份报告；人返回或断流使旧报告失效；回答绑定事件帧；保存实际显存峰值和检测 p99。

将命令、环境、实际输出、问题原因和你修改的代码写入 `runs/day25/notes.md`。未安装依赖、跳过、只读过源码与通过验收是四种不同状态。

遇到失败：先保留日志，再检查先修日的产物与[环境搭建](../../环境搭建.md)。平台限制见[架构与验收](../../架构与验收.md)。
