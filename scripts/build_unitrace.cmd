@echo off
setlocal
set PYTHONUTF8=1
for %%I in ("%~dp0..\..\nemotron-sycl") do set "DEPLOY=%%~fI"
if not "%~1"=="" set "DEPLOY=%~1"
set "VSROOT=C:\Program Files\Microsoft Visual Studio\2022\Community"
call "%VSROOT%\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 exit /b %errorlevel%
call "%DEPLOY%\toolchain\oneapi\setvars.bat" intel64 --force
if errorlevel 1 exit /b %errorlevel%
set "PATH=%VSROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
set "CMAKE=%VSROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
"%CMAKE%" -S "%DEPLOY%\toolchain\pti-gpu\tools\unitrace" -B "%DEPLOY%\toolchain\build-unitrace" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl "-DPython_EXECUTABLE=%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe" -DBUILD_WITH_MPI=OFF -DBUILD_WITH_ITT=OFF -DBUILD_WITH_XPTI=OFF -DBUILD_WITH_OPENCL=OFF -DBUILD_WITH_OMP=OFF -DBUILD_WITH_PERFETTO=OFF
if errorlevel 1 exit /b %errorlevel%
"%CMAKE%" --build "%DEPLOY%\toolchain\build-unitrace" -j 4
exit /b %errorlevel%
