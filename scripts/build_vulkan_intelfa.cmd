@echo off
setlocal
rem Source trees and source-manifest.json prepared as documented in docs/qwen38-kernel-experiments.md.
if "%~1"=="" (echo Usage: build_vulkan_intelfa.cmd ABSOLUTE_PORTABLE_ROOT & exit /b 2)
set "VROOT=%~f1"
if not defined B60_VSROOT set "B60_VSROOT=C:\Program Files\Microsoft Visual Studio\2022\Community"
call "%B60_VSROOT%\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 exit /b %errorlevel%
set "PATH=%B60_VSROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
set "CMAKE=%B60_VSROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set PYTHONUTF8=1
"%CMAKE%" -S "%VROOT%\shaderc" -B "%VROOT%\build-shaderc" -G Ninja -DCMAKE_BUILD_TYPE=Release -DSHADERC_SKIP_TESTS=ON -DSHADERC_SKIP_EXAMPLES=ON -DSPIRV_SKIP_TESTS=ON -DSHADERC_SKIP_COPYRIGHT_CHECK=ON -DENABLE_GLSLANG_BINARIES=OFF
if errorlevel 1 exit /b %errorlevel%
"%CMAKE%" --build "%VROOT%\build-shaderc" --target glslc_exe -j 4
if errorlevel 1 exit /b %errorlevel%
dumpbin /exports C:\Windows\System32\vulkan-1.dll > "%VROOT%\loader-exports.txt"
python "%~dp0vulkan_import_def.py" "%VROOT%\loader-exports.txt" "%VROOT%\vulkan-1.def"
if errorlevel 1 exit /b %errorlevel%
lib /nologo /def:"%VROOT%\vulkan-1.def" /machine:x64 /out:"%VROOT%\vulkan-1.lib"
if errorlevel 1 exit /b %errorlevel%
"%CMAKE%" -S "%VROOT%\shaderc\third_party\spirv-headers" -B "%VROOT%\build-spv" -G Ninja -DCMAKE_INSTALL_PREFIX="%VROOT%\spv-install"
if errorlevel 1 exit /b %errorlevel%
"%CMAKE%" --install "%VROOT%\build-spv"
if errorlevel 1 exit /b %errorlevel%
set "INCLUDE=%VROOT%\spv-install\include;%INCLUDE%"
"%CMAKE%" -S "%VROOT%\llama-intelfa-b11869" -B "%VROOT%\build-llama" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DVulkan_INCLUDE_DIR="%VROOT%\Vulkan-Headers\include" -DVulkan_LIBRARY="%VROOT%\vulkan-1.lib" -DVulkan_GLSLC_EXECUTABLE="%VROOT%\build-shaderc\glslc\glslc.exe" -DCMAKE_PREFIX_PATH="%VROOT%\spv-install" -DGGML_NATIVE=ON -DLLAMA_BUILD_NUMBER=11869 -DLLAMA_BUILD_COMMIT=d327bf1-intelfa -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=ON
if errorlevel 1 exit /b %errorlevel%
"%CMAKE%" --build "%VROOT%\build-llama" --target llama-server llama-bench -j 4
exit /b %errorlevel%
