#!/usr/bin/env python3
"""
测试 Linux 环境下的 vanitygen-plusplus 设置
"""
import os
import sys
import subprocess

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.utils.vanitygen_plusplus import is_vpp_available, _find_all_exes

def test_setup():
    print("="*50)
    print("Vanity Service Linux 环境检查")
    print("="*50)
    
    # 1. 检查操作系统
    print(f"\n1. 操作系统: {os.name}")
    print(f"   平台: {sys.platform}")
    
    # 2. 检查 vanitygen-plusplus 可用性
    print(f"\n2. Vanitygen-plusplus 可用: {is_vpp_available()}")
    
    # 3. 列出找到的可执行文件
    exes = _find_all_exes()
    print(f"\n3. 找到的可执行文件 ({len(exes)} 个):")
    for exe in exes:
        # 判断是 CPU 还是 GPU 版本
        exe_type = "GPU" if "ocl" in os.path.basename(exe).lower() else "CPU"
        print(f"   - {exe} ({exe_type}版本)")
    
    # 4. 检查环境变量
    print("\n4. 环境变量:")
    print(f"   DEFAULT_TIMEOUT: {os.getenv('DEFAULT_TIMEOUT', '未设置')}")
    print(f"   VPP_DEBUG: {os.getenv('VPP_DEBUG', '未设置')}")
    print(f"   VPP_PLATFORM: {os.getenv('VPP_PLATFORM', '未设置')}")
    print(f"   VPP_DEVICE: {os.getenv('VPP_DEVICE', '未设置')}")
    
    # 5. 测试执行（如果有可执行文件）
    if exes:
        print("\n5. 测试执行:")
        test_exe = exes[0]
        print(f"   测试: {test_exe}")
        try:
            # 只运行 -h 查看帮助
            result = subprocess.run([test_exe, "-h"], 
                                  capture_output=True, 
                                  text=True,
                                  timeout=5)
            if result.returncode == 0:
                print("   ✓ 可执行文件正常工作")
                # 显示版本信息
                for line in result.stdout.split('\n'):
                    if 'version' in line.lower() or 'vanitygen' in line.lower():
                        print(f"   版本: {line.strip()}")
                        break
            else:
                print("   ✗ 执行失败")
        except Exception as e:
            print(f"   ✗ 测试失败: {e}")
    
    print("\n" + "="*50)
    
    # 提供建议
    if not is_vpp_available():
        print("\n建议:")
        print("1. 运行 ./build_vanitygen.sh 构建 vanitygen-plusplus")
        print("2. 或者运行 ./setup_linux.sh 进行完整设置")
    elif not any('ocl' in exe for exe in exes):
        print("\n注意: 只找到 CPU 版本，生成速度会较慢")
        print("如果服务器有 GPU，请安装 OpenCL 并重新构建")

if __name__ == "__main__":
    test_setup()
