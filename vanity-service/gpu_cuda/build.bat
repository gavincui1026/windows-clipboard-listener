@echo off
echo Building CUDA TRON GPU Generator...

REM 设置CUDA路径
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1

REM 编译CUDA代码
nvcc -O3 -arch=sm_89 -shared -o tron_gpu.dll tron_gpu_generator.cu

if %errorlevel% neq 0 (
    echo Build failed!
    pause
    exit /b 1
)

echo Build successful!
echo Output: tron_gpu.dll
pause
