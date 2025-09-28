"""
Python包装器 - 调用C++ CUDA生成器
"""
import ctypes
import os
import hashlib
import base58
import sys
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
        
        from ctypes import c_int, c_char_p, c_char, POINTER
        self.lib.generate_addresses_gpu.argtypes = [
            c_char_p,           # prefix
            c_char_p,           # suffix
            POINTER(c_char),    # out_address
            POINTER(c_char),    # out_private_key
            c_int               # max_attempts
        ]
        self.lib.generate_addresses_gpu.restype = ctypes.c_int
        
        # 初始化CUDA
        if self.lib.cuda_init() != 0:
            raise RuntimeError("CUDA初始化失败")
        
        print("✅ C++ CUDA生成器初始化成功")
    
    async def generate(self, pattern: str, timeout: float = 0.0) -> Optional[Dict]:
        """生成匹配模式的地址"""
        # 解析：支持两种输入 + 轻量校验（Base58字符 + 总长约束）
        # 1) 完整TRON地址: 仅使用后5位作为匹配
        # 2) 形如 "...YYYYY" 的模式: 直接把 suffix=YYYYY
        raw = pattern.strip()
        prefix = ""
        suffix = ""
        if '...' in raw:
            parts = raw.split('...')
            # 新规则：忽略显式前缀，仅使用后缀
            suffix = parts[1] if len(parts) > 1 else parts[0]
        elif raw.startswith('T') and len(raw) == 34:
            suffix = raw[-5:]
        else:
            # 仅后缀
            suffix = raw
        # 校验允许字符与总长度（T + 33）
        b58chars = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
        if any(c not in b58chars for c in prefix+suffix):
            print("⚠️ 非法字符：仅允许Base58字符集")
            return None
        if len(prefix) + len(suffix) > 33:
            print("⚠️ 模式过长：prefix+suffix 之和不能超过33")
            return None
        
        # 准备输出缓冲区
        address_buffer = ctypes.create_string_buffer(35)
        private_key_buffer = ctypes.create_string_buffer(65)
        
        # 调试：初始化缓冲区
        address_buffer.value = b'\0' * 34
        private_key_buffer.value = b'\0' * 64
        
        # 计算最大尝试次数（基于超时）
        # 限制在int32范围内，避免溢出
        # timeout<=0 表示不限次
        max_attempts = 0 if timeout <= 0 else min(int(timeout * 10_000_000), 2_000_000_000)
        
        print(f"🚀 C++ CUDA生成开始")
        print(f"   原始模式: {pattern}")
        print(f"   实际匹配: {prefix}...{suffix} (共{len(prefix)+len(suffix)}个字符)")
        print(f"   最大尝试: {max_attempts:,}")
        
        # 调用CUDA函数
        import time
        start_time = time.time()
        
        import asyncio
        result = await asyncio.to_thread(
            self.lib.generate_addresses_gpu,
            prefix.encode(),
            suffix.encode(),
            address_buffer,
            private_key_buffer,
            max_attempts
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n📊 C++ 返回值: {result}")
        print(f"   Buffer长度: addr={len(address_buffer.raw.rstrip(b'\0'))}, key={len(private_key_buffer.raw.rstrip(b'\0'))}")
        
        if result > 0:
            address = address_buffer.value.decode().rstrip('\0')
            private_key = private_key_buffer.value.decode().rstrip('\0')
            speed = result / elapsed
            
            print(f"\n🔍 调试信息:")
            print(f"   原始模式: {pattern}")
            print(f"   Buffer内容 (前20字节): {address_buffer.raw[:20]}")
            print(f"   解码地址: '{address}'")
            print(f"   地址长度: {len(address)}")
            
            # 校验Base58Check与模式
            def valid_tron(addr: str) -> bool:
                try:
                    raw25 = base58.b58decode(addr)
                except Exception:
                    return False
                if len(raw25) != 25 or raw25[0] != 0x41:
                    return False
                chk = hashlib.sha256(hashlib.sha256(raw25[:21]).digest()).digest()[:4]
                return raw25[21:] == chk

            def match(addr: str) -> bool:
                # 仅匹配后缀（后5位）
                if suffix:
                    return addr.endswith(suffix)
                return True

            if valid_tron(address) and match(address):
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
                print("⚠️ GPU返回地址校验失败或不匹配模式，回退CPU…")
                # CPU fallback
                try:
                    cpu_mod_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'generators')
                    sys.path.append(cpu_mod_path)
                    from tron_generator_fixed import generate_real_tron_vanity
                    # 构造目标地址
                    target = raw if (raw.startswith('T') and len(raw) == 34) else (
                        'T' + prefix + ('X' * max(0, 34 - 1 - len(prefix) - len(suffix))) + suffix
                    )
                    cpu_result = generate_real_tron_vanity(target, timeout=timeout)
                    if cpu_result and cpu_result.get('found'):
                        return {
                            'address': cpu_result['address'],
                            'private_key': cpu_result['private_key'],
                            'type': 'TRON',
                            'attempts': cpu_result.get('attempts', 0),
                            'time': cpu_result.get('time', elapsed),
                            'backend': 'CPU fallback'
                        }
                except Exception as _e:
                    print(f"CPU回退失败: {_e}")
                return None
        else:
            print(f"\n❌ 未找到匹配 (尝试了{max_attempts:,}次)")
            # CPU fallback
            try:
                cpu_mod_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'generators')
                sys.path.append(cpu_mod_path)
                from tron_generator_fixed import generate_real_tron_vanity
                target = raw if (raw.startswith('T') and len(raw) == 34) else (
                    'T' + prefix + ('X' * max(0, 34 - 1 - len(prefix) - len(suffix))) + suffix
                )
                cpu_result = generate_real_tron_vanity(target, timeout=timeout)
                if cpu_result and cpu_result.get('found'):
                    return {
                        'address': cpu_result['address'],
                        'private_key': cpu_result['private_key'],
                        'type': 'TRON',
                        'attempts': cpu_result.get('attempts', 0),
                        'time': cpu_result.get('time', 0.0),
                        'backend': 'CPU fallback'
                    }
            except Exception as _e:
                print(f"CPU回退失败: {_e}")
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
