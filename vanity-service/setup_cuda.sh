#!/bin/bash

echo ""
echo "============================================"
echo "设置C++ CUDA GPU生成器 (Linux)"
echo "============================================"
echo ""

# 检查是否为root用户
if [ "$EUID" -eq 0 ]; then 
   echo "警告: 不建议使用root用户运行"
fi

# 检查CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ 未检测到CUDA编译器 (nvcc)"
    echo ""
    echo "请安装CUDA Toolkit:"
    echo ""
    echo "Ubuntu/Debian:"
    echo "  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb"
    echo "  sudo dpkg -i cuda-keyring_1.0-1_all.deb"
    echo "  sudo apt-get update"
    echo "  sudo apt-get -y install cuda"
    echo ""
    echo "或访问: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

echo "✅ 检测到CUDA编译器"
nvcc --version

# 检查NVIDIA驱动
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  未检测到NVIDIA驱动"
    echo "   请安装NVIDIA驱动"
    exit 1
fi

echo ""
echo "GPU信息:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# 检查GCC
if ! command -v g++ &> /dev/null; then
    echo ""
    echo "❌ 未检测到g++编译器"
    echo "   请安装: sudo apt-get install build-essential"
    exit 1
fi

echo ""
echo "开始编译CUDA代码..."
cd gpu_cuda

# 设置执行权限
chmod +x build.sh

# 编译
./build.sh

if [ -f "tron_gpu.so" ]; then
    echo ""
    echo "✅ 编译成功！"
    echo ""
    echo "现在可以使用GPU生成器了:"
    echo "- 重启vanity-service"
    echo "- API会自动使用C++ CUDA加速"
    echo ""
    echo "预期性能: 1亿+地址/秒 (RTX 5070 Ti)"
    
    # 检查Python绑定
    echo ""
    echo "测试Python绑定..."
    python3 -c "import ctypes; lib = ctypes.CDLL('./tron_gpu.so', ctypes.RTLD_GLOBAL); print('✅ Python可以加载CUDA库')" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "⚠️  Python加载测试失败"
        echo "   可能需要设置LD_LIBRARY_PATH"
        echo "   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)"
    fi
else
    echo ""
    echo "❌ 编译失败"
    echo ""
    echo "可能的原因:"
    echo "1. CUDA版本不匹配"
    echo "2. 显卡架构不支持"
    echo "3. 缺少依赖库"
    echo ""
    echo "查看详细错误信息，运行:"
    echo "cd gpu_cuda && ./build.sh"
fi

cd ..
echo ""
