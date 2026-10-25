@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 exit /b %errorlevel%
call "C:\MyLLM\nemotron-sycl\toolchain\oneapi\setvars.bat" intel64 --force
if errorlevel 1 exit /b %errorlevel%
set "DNNLROOT=C:\MyLLM\nemotron-sycl\toolchain\oneapi\dnnl\latest"
icx -fsycl /O2 /std:c++17 /EHsc /MD "%~dp0..\tools\sdpa_layout_probe.cpp" -I"%DNNLROOT%\include" /link /LIBPATH:"%DNNLROOT%\lib" dnnl.lib /OUT:"%~dp0..\runs\sdpa-layout-probe.exe"
exit /b %errorlevel%
