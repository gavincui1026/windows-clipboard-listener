#!/bin/bash
# 比较 CPU 和 GPU 版本的性能

echo "======================================"
echo "CPU vs GPU 性能对比测试"
echo "======================================"

# 测试地址
TEST_ADDR="T3jWrs"

# 检查可执行文件
echo -e "\n检查可执行文件："
if [ -f "vanitygen-plusplus/oclvanitygen++" ]; then
    echo "✓ GPU版本: vanitygen-plusplus/oclvanitygen++"
    GPU_EXE="vanitygen-plusplus/oclvanitygen++"
else
    echo "✗ 未找到 GPU 版本"
fi

if [ -f "vanitygen-plusplus/vanitygen++" ]; then
    echo "✓ CPU版本: vanitygen-plusplus/vanitygen++"
    CPU_EXE="vanitygen-plusplus/vanitygen++"
else
    echo "✗ 未找到 CPU 版本"
fi

# 测试 GPU 版本
if [ ! -z "$GPU_EXE" ]; then
    echo -e "\n测试 GPU 版本："
    echo "命令: $GPU_EXE -q -z -1 -C TRX $TEST_ADDR"
    echo "开始时间: $(date +%s.%N)"
    START=$(date +%s.%N)
    $GPU_EXE -q -z -1 -C TRX $TEST_ADDR
    END=$(date +%s.%N)
    GPU_TIME=$(echo "$END - $START" | bc)
    echo "结束时间: $(date +%s.%N)"
    echo "GPU 耗时: $GPU_TIME 秒"
fi

# 测试 CPU 版本
if [ ! -z "$CPU_EXE" ]; then
    echo -e "\n测试 CPU 版本："
    echo "命令: $CPU_EXE -q -z -1 -C TRX $TEST_ADDR"
    echo "开始时间: $(date +%s.%N)"
    START=$(date +%s.%N)
    timeout 30 $CPU_EXE -q -z -1 -C TRX $TEST_ADDR
    END=$(date +%s.%N)
    CPU_TIME=$(echo "$END - $START" | bc)
    echo "结束时间: $(date +%s.%N)"
    echo "CPU 耗时: $CPU_TIME 秒"
fi

# 显示 OpenCL 信息
echo -e "\n检查 OpenCL 设备："
if command -v clinfo &> /dev/null; then
    clinfo -l
else
    echo "clinfo 未安装，无法查看 OpenCL 设备"
fi

echo -e "\n======================================"
