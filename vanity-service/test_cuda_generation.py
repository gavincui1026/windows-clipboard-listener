#!/usr/bin/env python3
"""
测试CUDA地址生成器
"""
import os
import sys
import ctypes
import time

# 添加gpu_cuda目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'gpu_cuda'))

def test_basic_generation():
    """测试基本的地址生成功能"""
    try:
        # 根据平台选择库文件
        import platform
        if platform.system() == 'Windows':
            lib_name = 'tron_gpu.dll'
        else:
            lib_name = 'tron_gpu.so'
            
        lib_path = os.path.join(os.path.dirname(__file__), 'gpu_cuda', lib_name)
        print(f"Loading library: {lib_path}")
        
        if not os.path.exists(lib_path):
            print(f"错误: 找不到库文件 {lib_path}")
            return
            
        # 加载库
        if platform.system() != 'Windows':
            lib = ctypes.CDLL(lib_path, ctypes.RTLD_GLOBAL)
        else:
            lib = ctypes.CDLL(lib_path)
            
        # 定义函数接口
        lib.cuda_init.restype = ctypes.c_int
        lib.generate_addresses_gpu.argtypes = [
            ctypes.c_char_p,  # prefix
            ctypes.c_char_p,  # suffix
            ctypes.c_char_p,  # out_address
            ctypes.c_char_p,  # out_private_key
            ctypes.c_int      # max_attempts
        ]
        lib.generate_addresses_gpu.restype = ctypes.c_int
        
        # 初始化CUDA
        print("\n初始化CUDA...")
        if lib.cuda_init() != 0:
            print("CUDA初始化失败！")
            return
            
        # 测试1：生成任意地址（空模式）
        print("\n测试1: 生成任意TRON地址")
        address_buffer = ctypes.create_string_buffer(35)
        private_key_buffer = ctypes.create_string_buffer(65)
        
        result = lib.generate_addresses_gpu(
            b"",  # 空前缀
            b"",  # 空后缀
            address_buffer,
            private_key_buffer,
            1000  # 只尝试1000次
        )
        
        if result > 0:
            address = address_buffer.value.decode()
            print(f"✅ 生成地址: {address}")
            print(f"   长度: {len(address)}")
            print(f"   是否以T开头: {address.startswith('T')}")
            print(f"   私钥前32位: {private_key_buffer.value.decode()[:32]}")
        else:
            print("❌ 生成失败")
            
        # 测试2：生成简单模式
        print("\n测试2: 生成带模式的地址 (前缀='KZ', 后缀='')")
        address_buffer2 = ctypes.create_string_buffer(35)
        private_key_buffer2 = ctypes.create_string_buffer(65)
        
        start_time = time.time()
        result2 = lib.generate_addresses_gpu(
            b"KZ",  # 前缀（不含T）
            b"",    # 空后缀
            address_buffer2,
            private_key_buffer2,
            10_000_000  # 1000万次
        )
        elapsed = time.time() - start_time
        
        if result2 > 0:
            address2 = address_buffer2.value.decode()
            print(f"✅ 生成地址: {address2}")
            print(f"   匹配检查: T{address2[1:3]} == TKZ ? {address2[1:3] == 'KZ'}")
            print(f"   尝试次数: {result2:,}")
            print(f"   耗时: {elapsed:.3f}秒")
            print(f"   速度: {result2/elapsed:,.0f}次/秒")
        else:
            print(f"❌ 未找到匹配 (耗时{elapsed:.3f}秒)")
            
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_generation()
