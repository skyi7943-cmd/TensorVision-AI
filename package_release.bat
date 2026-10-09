@echo off
setlocal
cd /d "%~dp0"

echo ========================================================
echo   TensorVision AI - Release Packaging Tool
echo ========================================================

if not exist "dist\TensorVisionAI\TensorVisionAI.exe" (
    echo [!] dist\TensorVisionAI\TensorVisionAI.exe not found!
    echo [!] Running build_exe.py first...
    python build_exe.py
)

echo [*] Compressing dist\TensorVisionAI into TensorVisionAI-v2.0-Win64.zip ...
echo [*] This may take 1-2 minutes due to CUDA 12.4 DLLs (~4GB)...

tar -a -c -f TensorVisionAI-v2.0-Win64.zip -C dist TensorVisionAI

if exist "TensorVisionAI-v2.0-Win64.zip" (
    echo ========================================================
    echo [SUCCESS] Package created: TensorVisionAI-v2.0-Win64.zip
    echo [INFO] You can now upload this zip file directly to your
    echo        GitHub repository's Releases page (Release v2.0.0)!
    echo ========================================================
) else (
    echo [ERROR] Failed to create zip package.
)

pause
