@echo off
echo.
echo ============================================
echo 跨平台GPU加速工具安装脚本 (Windows)
echo ============================================
echo.

REM 检查Python版本
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo [1/3] 检测GPU环境...
python install_gpu.py

echo.
echo [2/3] 安装GPU加速库...
echo.

REM 检测NVIDIA GPU
nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    echo 检测到NVIDIA GPU，安装CUDA加速库...
    
    REM 检测CUDA版本
    nvcc --version | findstr "release 12" >nul 2>&1
    if %errorlevel%==0 (
        echo 安装CuPy for CUDA 12.x...
        pip install cupy-cuda12x==13.0.0
    ) else (
        nvcc --version | findstr "release 11" >nul 2>&1
        if %errorlevel%==0 (
            echo 安装CuPy for CUDA 11.x...
            pip install cupy-cuda11x==13.0.0
        ) else (
            echo 未检测到CUDA，安装默认版本...
            pip install cupy-cuda12x==13.0.0
        )
    )
    
    echo 安装Numba CUDA支持...
    pip install numba==0.60.0
)

echo.
echo 安装跨平台GPU库 (支持AMD/Intel)...
pip install pyopencl==2024.2.7
pip install numpy==1.26.4

echo.
echo [3/3] 测试GPU功能...
python -c "from app.generators.gpu_universal import get_gpu_info; print(get_gpu_info())"

echo.
echo ============================================
echo 安装完成！
echo.
echo 现在可以运行: python main.py
echo 服务将自动使用可用的GPU加速
echo ============================================
echo.
pause
