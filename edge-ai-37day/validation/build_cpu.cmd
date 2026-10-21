@echo off
rem Local Windows verification only; normal course builds use CMake on Linux.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
set "EDGE_PROJECT=%~dp0.."
pushd "%EDGE_PROJECT%\build\cpu"
cl /nologo /std:c++17 /EHsc /utf-8 /O1 /I"%EDGE_PROJECT%\pc\include" "%EDGE_PROJECT%\pc\src\core_tests.cpp" "%EDGE_PROJECT%\pc\src\cpu.cpp" /Fe:core_tests.exe
if errorlevel 1 exit /b 1
cl /nologo /std:c++17 /EHsc /utf-8 /O1 /I"%EDGE_PROJECT%\pc\include" "%EDGE_PROJECT%\pc\src\main.cpp" "%EDGE_PROJECT%\pc\src\cpu.cpp" /Fe:edge_pipeline.exe
if errorlevel 1 exit /b 1
core_tests.exe
if errorlevel 1 exit /b 1
popd
