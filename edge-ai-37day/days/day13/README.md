# Day 13 · GPU 预处理正确性

> 旧版实验入口，编号不对应新版时间线。请先按 [6个大任务计划](../../每日开发路线.md) 选择需要复用的实验；本目录run.py仍执行原有流程。


阶段：CUDA 与 TensorRT。先修：12；02。

## 今日学习与实现

需要掌握：双线性拉伸、半像素采样、RGB/NCHW、归一化。

用六张 fixture 比较 CPU 参考和 CUDA RGB→NCHW 预处理；检查奇数 padding 与不同宽高比。

本目录的 `run.py` 是当天可执行实验入口；具体实现已写在下面列出的共享源文件中。先运行，再逐段阅读，最后完成上面的参数对比/修改任务。每日共享实现是为了让你维护同一套完整系统，而不是复制 37 份不一致的代码。

## 运行

在项目根目录、已激活的 WSL Python 环境中执行（板端阶段改为板上终端）：

```bash
python days/day13/run.py
```



正常运行结束应返回 0；故障实验的预期错误由课程入口捕获并核对。

运行参数会保存到 `runs/day13/invocation.json`。实际实验数据写入该目录，模型共享产物位于 `artifacts/`，图片位于 `data/`；具体命令也会打印。自动脚本返回成功仅表示命令成功，下面的人工验收仍需完成。

## 要读和修改的代码

- [pc/src/cuda_backend.cu](../../pc/src/cuda_backend.cu)
- [pc/include/edge.hpp](../../pc/include/edge.hpp)
- [pc/src/cpu.cpp](../../pc/src/cpu.cpp)
- [tools/assets.py](../../tools/assets.py)
- [tools/course.py](../../tools/course.py)

`tools/course.py` 中 `day==13`（板端还有嵌套分支）给出全部实际调用，不会缺依赖后悄悄改用模拟模型。

## 验收

preprocess_max_error≤0.002；失败先查坐标/颜色/归一化，不能通过放宽阈值掩盖错误。

将命令、环境、实际输出、问题原因和你修改的代码写入 `runs/day13/notes.md`。未安装依赖、跳过、只读过源码与通过验收是四种不同状态。

遇到失败：先保留日志，再检查先修日的产物与[环境搭建](../../环境搭建.md)。平台限制见[架构与验收](../../架构与验收.md)。
