# 验证证据目录

- `detr-workstation/`：2026-10-14 RT-DETRv2-S 与工位巡检改动的实际CPU、规则和静态检查结果。
- `verify_local.py`：重跑本机检查，需要先编译CPU程序。
- `build_cpu.cmd`：本机已有MSVC路径的Windows直接编译脚本；正式课程使用Linux/CMake。
- `legacy-20260927.json` 及根目录原有 `.log` / `.json`：旧版本的历史证据，不能证明本次改动已验收。

范围和未测试项见 [验证说明](../验证说明.md)。
