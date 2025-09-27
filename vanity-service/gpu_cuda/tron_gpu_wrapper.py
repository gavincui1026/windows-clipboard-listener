"""
Python包装器 - 调用C++ CUDA生成器
"""
import ctypes
import os
from typing import Optional, Dict


class TronGPUGenerator:
    """CUDA GPU生成器包装器"""
    
    def __init__(self):
        # 根据平台加载库
        import platform
        
        if platform.system() == 'Windows':
            lib_name = 'tron_gpu.dll'
        else:  # Linux/Unix
            lib_name = 'tron_gpu.so'
        
        lib_path = os.path.join(os.path.dirname(__file__), lib_name)
        if not os.path.exists(lib_path):
            raise RuntimeError(f"找不到CUDA库: {lib_path}")
        
        # Linux需要RTLD_GLOBAL标志以正确加载CUDA符号
        if platform.system() != 'Windows':
            self.lib = ctypes.CDLL(lib_path, ctypes.RTLD_GLOBAL)
        else:
            self.lib = ctypes.CDLL(lib_path)
        
        # 定义函数接口
        self.lib.cuda_init.restype = ctypes.c_int
        
        self.lib.generate_addresses_gpu.argtypes = [
            ctypes.c_char_p,  # prefix
            ctypes.c_char_p,  # suffix
            ctypes.c_char_p,  # out_address
            ctypes.c_char_p,  # out_private_key
            ctypes.c_int      # max_attempts
        ]
        self.lib.generate_addresses_gpu.restype = ctypes.c_int
        
        # 初始化CUDA
        if self.lib.cuda_init() != 0:
            raise RuntimeError("CUDA初始化失败")
        
        print("✅ C++ CUDA生成器初始化成功")
    
    async def generate(self, pattern: str, timeout: float = 60.0) -> Optional[Dict]:
        """生成匹配模式的地址"""
        # 解析模式
        if '...' in pattern:
            parts = pattern.split('...')
            prefix = parts[0]
            suffix = parts[1] if len(parts) > 1 else ""
        else:
            prefix = pattern
            suffix = ""
        
        # 准备输出缓冲区
        address_buffer = ctypes.create_string_buffer(35)
        private_key_buffer = ctypes.create_string_buffer(65)
        
        # 计算最大尝试次数（基于超时）
        max_attempts = int(timeout * 10_000_000)  # 假设1000万/秒
        
        print(f"🚀 C++ CUDA生成开始")
        print(f"   模式: {prefix}...{suffix}")
        print(f"   最大尝试: {max_attempts:,}")
        
        # 调用CUDA函数
        import time
        start_time = time.time()
        
        result = self.lib.generate_addresses_gpu(
            prefix.encode(),
            suffix.encode(),
            address_buffer,
            private_key_buffer,
            max_attempts
        )
        
        elapsed = time.time() - start_time
        
        if result > 0:
            address = address_buffer.value.decode()
            private_key = private_key_buffer.value.decode()
            speed = result / elapsed
            
            print(f"\n✅ C++ CUDA找到匹配!")
            print(f"   地址: {address}")
            print(f"   私钥: {private_key[:32]}...")
            print(f"   尝试: {result:,}")
            print(f"   耗时: {elapsed:.3f}秒")
            print(f"   速度: {speed:,.0f}/秒")
            
            return {
                'address': address,
                'private_key': private_key,
                'type': 'TRON',
                'attempts': result,
                'time': elapsed,
                'speed': speed,
                'backend': 'C++ CUDA (RTX 5070 Ti)'
            }
        else:
            print(f"\n❌ 未找到匹配 (尝试了{max_attempts:,}次)")
            return None


# 全局实例
cuda_generator = None
try:
    cuda_generator = TronGPUGenerator()
except Exception as e:
    print(f"CUDA生成器初始化失败: {e}")


async def generate_tron_cuda(address: str, timeout: float = 60.0) -> Optional[Dict]:
    """使用CUDA生成TRON地址"""
    if not cuda_generator:
        return None
    
    return await cuda_generator.generate(address, timeout)
