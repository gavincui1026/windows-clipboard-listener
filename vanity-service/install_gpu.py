#!/usr/bin/env python3
"""
跨平台GPU加速库安装脚本
自动检测系统和GPU类型，安装合适的加速库
"""
import os
import sys
import subprocess
import platform


def detect_gpu():
    """检测GPU类型"""
    gpu_info = {
        'nvidia': False,
        'amd': False,
        'intel': False,
        'cuda_version': None
    }
    
    # 检测NVIDIA GPU
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info['nvidia'] = True
            print("✓ 检测到NVIDIA GPU")
            
            # 检测CUDA版本
            cuda_result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            if cuda_result.returncode == 0:
                output = cuda_result.stdout
                if 'release 12' in output:
                    gpu_info['cuda_version'] = '12.x'
                elif 'release 11' in output:
                    gpu_info['cuda_version'] = '11.x'
                print(f"  CUDA版本: {gpu_info['cuda_version']}")
    except:
        pass
    
    # 检测AMD GPU (Windows)
    if platform.system() == 'Windows':
        try:
            result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                  capture_output=True, text=True)
            if 'AMD' in result.stdout or 'Radeon' in result.stdout:
                gpu_info['amd'] = True
                print("✓ 检测到AMD GPU")
        except:
            pass
    
    # 检测Intel GPU
    try:
        if platform.system() == 'Windows':
            result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                  capture_output=True, text=True)
            if 'Intel' in result.stdout:
                gpu_info['intel'] = True
                print("✓ 检测到Intel GPU")
    except:
        pass
    
    return gpu_info


def install_gpu_libs(gpu_info):
    """根据GPU类型安装合适的库"""
    packages = []
    
    # 基础包
    packages.append('numpy==1.26.4')
    
    if gpu_info['nvidia']:
        print("\n安装NVIDIA GPU加速库...")
        if gpu_info['cuda_version'] == '12.x':
            packages.append('cupy-cuda12x==13.0.0')
            print("  - CuPy (CUDA 12.x)")
        elif gpu_info['cuda_version'] == '11.x':
            packages.append('cupy-cuda11x==13.0.0')
            print("  - CuPy (CUDA 11.x)")
        else:
            # 默认安装最新版
            packages.append('cupy-cuda12x==13.0.0')
            print("  - CuPy (默认CUDA 12.x)")
        
        packages.append('numba==0.60.0')
        print("  - Numba (CUDA JIT编译)")
    
    # 安装跨平台OpenCL（支持所有GPU）
    print("\n安装跨平台GPU库...")
    packages.append('pyopencl==2024.2.7')
    print("  - PyOpenCL (支持NVIDIA/AMD/Intel)")
    
    # 执行安装
    print(f"\n执行安装命令...")
    for package in packages:
        print(f"安装: {package}")
        subprocess.run([sys.executable, '-m', 'pip', 'install', package])


def test_gpu_libs():
    """测试GPU库是否正常工作"""
    print("\n测试GPU加速库...")
    
    # 测试CuPy
    try:
        import cupy as cp
        print("\n✓ CuPy测试:")
        a = cp.array([1, 2, 3])
        b = cp.array([4, 5, 6])
        c = a + b
        print(f"  GPU计算: {a.get()} + {b.get()} = {c.get()}")
        print(f"  GPU设备: {cp.cuda.runtime.getDevice()}")
    except Exception as e:
        print(f"✗ CuPy不可用: {e}")
    
    # 测试PyOpenCL
    try:
        import pyopencl as cl
        print("\n✓ PyOpenCL测试:")
        platforms = cl.get_platforms()
        for i, platform in enumerate(platforms):
            print(f"  平台{i}: {platform.name}")
            devices = platform.get_devices()
            for j, device in enumerate(devices):
                print(f"    设备{j}: {device.name}")
    except Exception as e:
        print(f"✗ PyOpenCL不可用: {e}")
    
    # 测试Numba
    try:
        from numba import cuda
        print("\n✓ Numba CUDA测试:")
        if cuda.is_available():
            print(f"  CUDA可用")
            gpu = cuda.get_current_device()
            print(f"  GPU名称: {gpu.name}")
        else:
            print("  CUDA不可用")
    except Exception as e:
        print(f"✗ Numba不可用: {e}")


def main():
    print("=" * 60)
    print("跨平台GPU加速库安装器")
    print("=" * 60)
    
    # 检测系统
    print(f"\n系统信息:")
    print(f"  操作系统: {platform.system()} {platform.release()}")
    print(f"  Python版本: {sys.version.split()[0]}")
    
    # 检测GPU
    print(f"\n检测GPU...")
    gpu_info = detect_gpu()
    
    if not any([gpu_info['nvidia'], gpu_info['amd'], gpu_info['intel']]):
        print("\n⚠️ 未检测到GPU，将只能使用CPU模式")
        print("如果你确实有GPU，请确保:")
        print("  - NVIDIA: 安装了最新的NVIDIA驱动")
        print("  - AMD: 安装了最新的AMD驱动")
        print("  - Intel: 安装了最新的Intel Graphics驱动")
        
        response = input("\n是否继续安装? (y/n): ")
        if response.lower() != 'y':
            return
    
    # 安装库
    install_gpu_libs(gpu_info)
    
    # 测试
    print("\n" + "=" * 60)
    test_gpu_libs()
    
    print("\n" + "=" * 60)
    print("安装完成！")
    print("\n使用说明:")
    print("1. CuPy - 用于NVIDIA GPU的NumPy兼容库")
    print("   import cupy as cp")
    print("   # 使用方法类似NumPy，但在GPU上运行")
    print("")
    print("2. PyOpenCL - 跨平台GPU编程")
    print("   import pyopencl as cl")
    print("   # 支持NVIDIA/AMD/Intel GPU")
    print("")
    print("3. Numba - JIT编译和CUDA支持")
    print("   from numba import cuda, jit")
    print("   # 可以将Python函数编译为GPU代码")


if __name__ == "__main__":
    main()
