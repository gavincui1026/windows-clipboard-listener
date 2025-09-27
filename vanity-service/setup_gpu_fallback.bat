@echo off
echo.
echo ============================================
echo GPU加速备用方案 - 使用PyOpenCL
echo ============================================
echo.
echo 由于CUDA问题，将使用PyOpenCL作为备用方案
echo PyOpenCL支持NVIDIA/AMD/Intel GPU
echo.

REM 安装PyOpenCL
echo [1] 安装PyOpenCL...
pip install pyopencl==2024.2.7
pip install numpy==1.26.4

echo.
echo [2] 安装Numba (CPU加速)...
pip install numba==0.60.0

echo.
echo [3] 测试PyOpenCL...
python -c "import pyopencl as cl; platforms=cl.get_platforms(); print(f'OpenCL平台数: {len(platforms)}'); [print(f'平台: {p.name}') for p in platforms]"

echo.
echo [4] 创建配置文件...
echo # GPU配置 > gpu_config.txt
echo USE_CUPY=false >> gpu_config.txt
echo USE_OPENCL=true >> gpu_config.txt
echo USE_NUMBA=true >> gpu_config.txt

echo.
echo ============================================
echo 备用方案安装完成！
echo.
echo 现在服务将优先使用PyOpenCL进行GPU加速
echo 如果OpenCL也不可用，将使用Numba加速CPU计算
echo ============================================
echo.
pause
