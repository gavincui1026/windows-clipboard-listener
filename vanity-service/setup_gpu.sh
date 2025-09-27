#!/bin/bash

echo ""
echo "============================================"
echo "跨平台GPU加速工具安装脚本 (Linux/Mac)"
echo "============================================"
echo ""

# 检查Python版本
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python 3.8+"
    exit 1
fi

echo "[1/3] 检测GPU环境..."
python3 install_gpu.py

echo ""
echo "[2/3] 安装GPU加速库..."
echo ""

# 检测NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "检测到NVIDIA GPU，安装CUDA加速库..."
    
    # 检测CUDA版本
    if nvcc --version | grep -q "release 12"; then
        echo "安装CuPy for CUDA 12.x..."
        pip3 install cupy-cuda12x==13.0.0
    elif nvcc --version | grep -q "release 11"; then
        echo "安装CuPy for CUDA 11.x..."
        pip3 install cupy-cuda11x==13.0.0
    else
        echo "未检测到CUDA，安装默认版本..."
        pip3 install cupy-cuda12x==13.0.0
    fi
    
    echo "安装Numba CUDA支持..."
    pip3 install numba==0.60.0
fi

echo ""
echo "安装跨平台GPU库 (支持AMD/Intel)..."
pip3 install pyopencl==2024.2.7
pip3 install numpy==1.26.4

# Linux可能需要安装OpenCL驱动
if [ "$(uname)" == "Linux" ]; then
    echo ""
    echo "提示: Linux系统可能需要安装OpenCL驱动:"
    echo "  NVIDIA: 已包含在驱动中"
    echo "  AMD: sudo apt install rocm-opencl-runtime"
    echo "  Intel: sudo apt install intel-opencl-icd"
fi

echo ""
echo "[3/3] 测试GPU功能..."
python3 -c "from app.generators.gpu_universal import get_gpu_info; print(get_gpu_info())"

echo ""
echo "============================================"
echo "安装完成！"
echo ""
echo "现在可以运行: python3 main.py"
echo "服务将自动使用可用的GPU加速"
echo "============================================"
echo ""
