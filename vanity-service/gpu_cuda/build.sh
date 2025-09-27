#!/bin/bash

echo "Building CUDA TRON GPU Generator for Linux..."

# 检查CUDA
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA compiler (nvcc) not found!"
    echo "Please install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# 获取CUDA版本和GPU架构
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
echo "CUDA Version: $CUDA_VERSION"

# 自动检测GPU架构
GPU_ARCH=""
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    echo "GPU: $GPU_NAME"
    
    # 根据GPU型号设置架构
    case "$GPU_NAME" in
        *"RTX 5070"*|*"RTX 5080"*|*"RTX 5090"*)
            GPU_ARCH="sm_90"  # Ada Lovelace
            ;;
        *"RTX 4090"*|*"RTX 4080"*|*"RTX 4070"*)
            GPU_ARCH="sm_89"  # Ada Lovelace
            ;;
        *"RTX 3090"*|*"RTX 3080"*|*"RTX 3070"*)
            GPU_ARCH="sm_86"  # Ampere
            ;;
        *"RTX 2080"*|*"RTX 2070"*)
            GPU_ARCH="sm_75"  # Turing
            ;;
        *)
            GPU_ARCH="sm_70"  # 默认Pascal+
            ;;
    esac
else
    GPU_ARCH="sm_70"
fi

echo "GPU Architecture: $GPU_ARCH"

# 编译命令
echo "Compiling..."
nvcc -O3 -arch=$GPU_ARCH -Xcompiler -fPIC -shared -o tron_gpu.so tron_gpu_generator.cu

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Output: tron_gpu.so"
    
    # 设置权限
    chmod +x tron_gpu.so
    
    # 显示依赖信息
    echo ""
    echo "Library dependencies:"
    ldd tron_gpu.so | grep -E "(cuda|cudart)"
else
    echo "Build failed!"
    exit 1
fi
