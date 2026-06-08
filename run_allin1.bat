@echo off
setlocal

set "ROOT=%~dp0"
set "MPLCONFIGDIR=%ROOT%.cache\matplotlib"
set "HF_HOME=%ROOT%.cache\huggingface"
set "TORCH_HOME=%ROOT%.cache\torch"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%"
if not exist "%HF_HOME%" mkdir "%HF_HOME%"
if not exist "%TORCH_HOME%" mkdir "%TORCH_HOME%"

"%ROOT%.venv\Scripts\allin1.exe" %*
exit /b %ERRORLEVEL%
