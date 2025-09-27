"""
快速测试GPU是否可用
"""
import subprocess
import sys

print("检测GPU环境...")
print("=" * 60)

# 1. 检查NVIDIA驱动
print("\n[1] 检查NVIDIA驱动")
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ GPU: {result.stdout.strip()}")
    else:
        print("✗ nvidia-smi未找到")
except:
    print("✗ NVIDIA驱动未安装")

# 2. 检查CUDA
print("\n[2] 检查CUDA")
try:
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        print(f"✓ {lines[-1]}")
    else:
        print("✗ CUDA未安装")
        print("  下载: https://developer.nvidia.com/cuda-downloads")
except:
    print("✗ nvcc未找到")

# 3. 安装CuPy
print("\n[3] 安装GPU加速库")
print("运行以下命令安装CuPy:")
print("\n  pip install cupy-cuda12x")

print("\n[4] 测试代码")
print("""
# test_gpu.py
import cupy as cp
import time

# GPU测试
print("GPU测试:")
start = time.time()
a = cp.random.random((10000, 10000))
b = cp.random.random((10000, 10000))
c = cp.dot(a, b)
cp.cuda.Stream.null.synchronize()
print(f"矩阵乘法耗时: {time.time()-start:.3f}秒")

# CPU对比
import numpy as np
print("\\nCPU测试:")
start = time.time()
a = np.random.random((10000, 10000))
b = np.random.random((10000, 10000))
c = np.dot(a, b)
print(f"矩阵乘法耗时: {time.time()-start:.3f}秒")
""")

print("\n" + "=" * 60)
print("建议：")
print("1. 先用pip install cupy-cuda12x")
print("2. 如果需要更高性能，安装CUDA Toolkit")
print("3. RTX 4070可以提供100倍以上加速！")
