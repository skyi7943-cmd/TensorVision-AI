@echo off
setlocal
cd /d "%~dp0"

if exist "%~dp0dist\TensorVisionAI\TensorVisionAI.exe" (
    start "" "%~dp0dist\TensorVisionAI\TensorVisionAI.exe"
    exit /b 0
)

if exist "E:\Python312\python.exe" (
    "E:\Python312\python.exe" main.py
) else (
    python main.py
)

if errorlevel 1 pause
