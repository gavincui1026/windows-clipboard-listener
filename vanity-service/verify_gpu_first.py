#!/usr/bin/env python3
"""
验证是否优先使用 GPU 版本
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.utils.vanitygen_plusplus import _find_all_exes

print("验证 vanitygen-plusplus 选择优先级")
print("="*50)

# 获取所有可执行文件
exes = _find_all_exes()

if not exes:
    print("✗ 没有找到任何可执行文件！")
    print("\n请先运行：")
    print("  ./build_vanitygen.sh  # 构建可执行文件")
    print("  或")
    print("  ./setup_linux.sh      # 完整设置")
else:
    print(f"找到 {len(exes)} 个可执行文件：\n")
    
    # 显示优先级顺序
    for i, exe in enumerate(exes):
        exe_name = os.path.basename(exe)
        exe_type = "GPU" if "ocl" in exe_name else "CPU"
        
        if i == 0:
            print(f"[优先使用] {exe}")
            print(f"           类型: {exe_type} 版本")
            print(f"           名称: {exe_name}")
            if exe_type == "CPU":
                print("           ⚠️  警告: 将使用 CPU 版本，速度会较慢！")
            else:
                print("           ✓ 很好: 将使用 GPU 加速版本")
        else:
            print(f"[备选 {i}]   {exe} ({exe_type})")
    
    print("\n" + "="*50)
    
    # 检查顺序是否正确
    first_exe = os.path.basename(exes[0])
    if "ocl" not in first_exe:
        print("\n⚠️  注意：当前会优先使用 CPU 版本！")
        print("可能的解决方案：")
        print("1. 确保已构建 GPU 版本（oclvanitygen++）")
        print("2. 检查文件权限")
        print("3. 临时禁用 CPU 版本：")
        print(f"   mv {exes[0]} {exes[0]}.bak")
    else:
        print("\n✓ 配置正确：将优先使用 GPU 加速版本")

# 显示操作系统信息
print(f"\n操作系统: {'Windows' if os.name == 'nt' else 'Linux/Unix'}")
print(f"Python 版本: {sys.version.split()[0]}")
