# Day 27 · 系统回归与故障规则

> 旧版实验入口，编号不对应新版时间线。请先按 [6个大任务计划](../../每日开发路线.md) 选择需要复用的实验；本目录run.py仍执行原有流程。


阶段：PC 验收。先修：26。

## 今日学习与实现

需要掌握：不变量、输入验证、回归测试。

运行项目测试、ROS 独立规则及巡检协调器回调测试、C++ 核心测试；核对断流、返回、忙状态、固定快照、超时和退出。

本目录的 `run.py` 是当天可执行实验入口；具体实现已写在下面列出的共享源文件中。先运行，再逐段阅读，最后完成上面的参数对比/修改任务。每日共享实现是为了让你维护同一套完整系统，而不是复制 37 份不一致的代码。

## 运行

在项目根目录、已激活的 WSL Python 环境中执行（板端阶段改为板上终端）：

```bash
python days/day27/run.py
```



正常运行结束应返回 0；故障实验的预期错误由课程入口捕获并核对。

运行参数会保存到 `runs/day27/invocation.json`。实际实验数据写入该目录，模型共享产物位于 `artifacts/`，图片位于 `data/`；具体命令也会打印。自动脚本返回成功仅表示命令成功，下面的人工验收仍需完成。

## 要读和修改的代码

- [tests/test_project.py](../../tests/test_project.py)
- [ros_ws/tests/test_core.py](../../ros_ws/tests/test_core.py)
- [pc/src/core_tests.cpp](../../pc/src/core_tests.cpp)
- [tools/course.py](../../tools/course.py)
- [ros_ws/tests/test_workstation.py](../../ros_ws/tests/test_workstation.py)
- [ros_ws/tests/test_coordinator_inspection.py](../../ros_ws/tests/test_coordinator_inspection.py)

`tools/course.py` 中 `day==27`（板端还有嵌套分支）给出全部实际调用，不会缺依赖后悄悄改用模拟模型。

## 验收

所有具备依赖的测试通过；跳过项逐项注明；CPU 过载测试计数守恒；测试通过不等于板端通过。

将命令、环境、实际输出、问题原因和你修改的代码写入 `runs/day27/notes.md`。未安装依赖、跳过、只读过源码与通过验收是四种不同状态。

遇到失败：先保留日志，再检查先修日的产物与[环境搭建](../../环境搭建.md)。平台限制见[架构与验收](../../架构与验收.md)。
