#!/usr/bin/env python3
"""
测试 GPU 使用情况
"""
import os
import sys
import subprocess
import time
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.utils.vanitygen_plusplus import _find_all_exes, generate_trx_with_vpp, build_trx_pattern

async def test_generate():
    print("="*60)
    print("测试 vanitygen-plusplus GPU 使用情况")
    print("="*60)
    
    # 1. 显示找到的可执行文件
    exes = _find_all_exes()
    print("\n1. 找到的可执行文件：")
    for i, exe in enumerate(exes):
        exe_name = os.path.basename(exe)
        if 'ocl' in exe_name:
            print(f"   [{i}] {exe} (GPU版本)")
        else:
            print(f"   [{i}] {exe} (CPU版本)")
    
    if not exes:
        print("   ✗ 没有找到可执行文件！")
        return
    
    # 2. 显示第一个将被使用的文件
    print(f"\n2. 将使用的文件: {exes[0]}")
    print(f"   类型: {'GPU' if 'ocl' in os.path.basename(exes[0]) else 'CPU'}")
    
    # 3. 测试地址
    test_address = "T3jWrsnDs1hQQSsSr0ZJebmgCtkMHLeN20E90A123930786078"
    pattern = build_trx_pattern(test_address)
    print(f"\n3. 测试地址: {test_address}")
    print(f"   匹配模式: {pattern}")
    
    # 4. 直接执行命令测试
    print("\n4. 直接执行测试：")
    exe = exes[0]
    cmd = [exe, "-q", "-z", "-1", "-C", "TRX", pattern]
    print(f"   命令: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        # 设置环境变量
        env = os.environ.copy()
        env['VPP_DEBUG'] = '1'
        
        print("   执行中...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.path.dirname(exe),
            env=env
        )
        
        elapsed = time.time() - start_time
        print(f"   耗时: {elapsed:.2f} 秒")
        
        if result.returncode == 0:
            print("   ✓ 执行成功")
            # 解析输出
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.startswith('TRX,'):
                    parts = line.split(',')
                    if len(parts) >= 4:
                        print(f"   生成地址: {parts[2]}")
                        print(f"   私钥: {parts[3][:10]}...")
                        # 验证前缀
                        if parts[2].startswith(pattern):
                            print(f"   ✓ 前缀匹配正确！")
                        else:
                            print(f"   ✗ 前缀不匹配！期望: {pattern}, 实际: {parts[2][:len(pattern)]}")
        else:
            print(f"   ✗ 执行失败: {result.returncode}")
            if result.stderr:
                print(f"   错误: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("   ✗ 执行超时（30秒）")
    except Exception as e:
        print(f"   ✗ 执行异常: {e}")
    
    # 5. 测试 Python 函数
    print("\n5. 测试 Python 异步函数：")
    print("   调用 generate_trx_with_vpp()...")
    start_time = time.time()
    
    result = await generate_trx_with_vpp(test_address)
    elapsed = time.time() - start_time
    
    print(f"   耗时: {elapsed:.2f} 秒")
    if result:
        print(f"   ✓ 生成成功")
        print(f"   地址: {result['address']}")
        print(f"   私钥: {result['private_key'][:10]}...")
        if result['address'].startswith(pattern):
            print(f"   ✓ 前缀匹配正确！")
        else:
            print(f"   ✗ 前缀不匹配！")
    else:
        print("   ✗ 生成失败")
    
    # 6. 性能分析
    print("\n6. 性能分析：")
    print(f"   - 匹配 {len(pattern)} 位前缀")
    print(f"   - 理论组合数: 58^{len(pattern)-1} = {58**(len(pattern)-1):,}")
    if elapsed < 1:
        print("   - 生成速度极快，可能原因：")
        print("     1. GPU 性能强大（使用了 OpenCL 加速）")
        print("     2. 运气好，很快找到匹配")
        print("     3. 可能有多个 GPU 设备并行计算")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    asyncio.run(test_generate())
