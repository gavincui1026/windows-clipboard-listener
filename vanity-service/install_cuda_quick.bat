@echo off
echo.
echo ============================================
echo 快速安装CUDA运行时库
echo ============================================
echo.

REM 方案1: 使用conda安装CUDA运行时（最简单）
echo [方案1] 使用conda安装CUDA运行时...
echo.
echo 如果你有Anaconda/Miniconda，运行:
echo conda install cudatoolkit=11.8
echo.

REM 方案2: 下载预编译的PyOpenCL
echo [方案2] 安装预编译的PyOpenCL wheel...
echo.
echo 正在下载适用于Windows的PyOpenCL...
pip install --only-binary :all: pyopencl

REM 方案3: 使用PyTorch的CUDA
echo.
echo [方案3] 通过PyTorch安装CUDA库...
echo.
echo 安装包含CUDA的PyTorch（会自动安装CUDA运行时）:
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo.
echo [测试] 测试GPU加速...
python -c "import torch; print(f'PyTorch CUDA可用: {torch.cuda.is_available()}')"

echo.
echo ============================================
echo 安装完成！
echo.
echo 如果以上方案都不行，请：
echo 1. 下载CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
echo 2. 或使用最简单的方案：pip install tensorflow (自带CUDA)
echo ============================================
pause
