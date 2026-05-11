@echo off
SETLOCAL

rem 将命令行窗口的编码设置为 UTF-8，以便正确显示中文
rem 注意：这要求批处理文件本身也以 UTF-8 (带 BOM) 编码保存
chcp 65001 > nul
cd /d %~dp0

call .venv\Scripts\activate.bat

blenderproc run ./settings/binpicking/GenDataset.py --config=./settings/binpicking/config_sample.yaml
cmd /k echo.