@echo off
setlocal
set PYTHONUTF8=1
if not defined B60_PYTHON set "B60_PYTHON=python"
set "B60_DEPLOYMENT=%~1"
if not defined B60_DEPLOYMENT set "B60_DEPLOYMENT=%~dp0..\nemotron-sycl"
"%B60_PYTHON%" "%~dp0scripts\manage_server.py" restart --deployment-dir "%B60_DEPLOYMENT%" --preset "%~dp0presets\qwen38-recommended.json"
if errorlevel 1 echo Start failed. Inspect the server log above.
pause
