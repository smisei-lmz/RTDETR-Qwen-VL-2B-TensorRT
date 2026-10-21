# Day 18 · 拆解性能瓶颈

> 旧版实验入口，编号不对应新版时间线。请先按 [6个大任务计划](../../每日开发路线.md) 选择需要复用的实验；本目录run.py仍执行原有流程。


阶段：CUDA 与 TensorRT。先修：17。

## 今日学习与实现

需要掌握：事件耗时、host 延迟、p50/p95/p99。

输出各阶段耗时；可加 --profile 生成 Nsight timeline；区分 CPU 调度、H2D、预处理、推理和结果回收。

本目录的 `run.py` 是当天可执行实验入口；具体实现已写在下面列出的共享源文件中。先运行，再逐段阅读，最后完成上面的参数对比/修改任务。每日共享实现是为了让你维护同一套完整系统，而不是复制 37 份不一致的代码。

## 运行

在项目根目录、已激活的 WSL Python 环境中执行（板端阶段改为板上终端）：

```bash
python days/day18/run.py
```

已安装 Nsight Systems 时可加 `--profile`。

正常运行结束应返回 0；故障实验的预期错误由课程入口捕获并核对。

运行参数会保存到 `runs/day18/invocation.json`。实际实验数据写入该目录，模型共享产物位于 `artifacts/`，图片位于 `data/`；具体命令也会打印。自动脚本返回成功仅表示命令成功，下面的人工验收仍需完成。

## 要读和修改的代码

- [tools/analyze.py](../../tools/analyze.py)
- [tools/course.py](../../tools/course.py)
- [pc/src/cuda_backend.cu](../../pc/src/cuda_backend.cu)
- [pc/include/edge.hpp](../../pc/include/edge.hpp)

`tools/course.py` 中 `day==18`（板端还有嵌套分支）给出全部实际调用，不会缺依赖后悄悄改用模拟模型。

## 验收

latency.json 有样本数和分位数；注明预热；不用 CUDA kernel 时间冒充端到端时间；CPU/GPU 测量时钟不混减。

将命令、环境、实际输出、问题原因和你修改的代码写入 `runs/day18/notes.md`。未安装依赖、跳过、只读过源码与通过验收是四种不同状态。

遇到失败：先保留日志，再检查先修日的产物与[环境搭建](../../环境搭建.md)。平台限制见[架构与验收](../../架构与验收.md)。
