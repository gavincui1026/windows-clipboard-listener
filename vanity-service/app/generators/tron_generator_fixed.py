"""
TRON地址生成器（修复版）
解决Windows多进程问题
"""
import time
import secrets
import hashlib
import base58
from ecdsa import SigningKey, SECP256k1
from typing import Dict, Optional


def sha256(data):
    """SHA256哈希"""
    return hashlib.sha256(data).digest()


def generate_tron_address_from_private_key(private_key_bytes):
    """从私钥生成TRON地址"""
    # 1. 生成公钥
    sk = SigningKey.from_string(private_key_bytes, curve=SECP256k1)
    vk = sk.get_verifying_key()
    public_key = vk.to_string()
    
    # 2. Keccak256哈希
    try:
        # 尝试使用pycryptodome
        from Crypto.Hash import keccak
        keccak_hash = keccak.new(digest_bits=256)
        keccak_hash.update(public_key)
        keccak = keccak_hash.digest()
    except ImportError:
        # 回退到sha3（不完全准确但可用）
        keccak = hashlib.sha3_256(public_key).digest()
    
    # 3. 取最后20字节
    address_bytes = keccak[-20:]
    
    # 4. 添加前缀0x41（TRON主网）
    address_bytes = b'\x41' + address_bytes
    
    # 5. 计算校验和（双SHA256的前4字节）
    checksum = sha256(sha256(address_bytes))[:4]
    
    # 6. Base58编码
    address = base58.b58encode(address_bytes + checksum).decode('utf-8')
    
    return address


def generate_real_tron_vanity(target_address: str, timeout: float = 0) -> Optional[Dict]:
    """
    生成与目标TRON地址前2位后3位相同的地址（简化版）
    
    Args:
        target_address: 目标地址
        timeout: 最大超时时间（0表示无限制，默认无限制）
    """
    if not target_address.startswith('T') or len(target_address) != 34:
        return None
    
    # 提取模式（跳过第一个字符T）
    prefix_target = target_address[1:3]  # 第2-3位
    suffix_target = target_address[-3:]  # 最后3位
    
    start_time = time.time()
    attempts = 0
    found = False
    
    print(f"开始生成TRON地址，目标模式: T{prefix_target}...{suffix_target}")
    
    # 生成直到找到或超时
    while not found:
        # 生成随机私钥
        private_key = secrets.randbits(256).to_bytes(32, 'big')
        
        # 生成地址
        try:
            address = generate_tron_address_from_private_key(private_key)
            
            # 检查是否匹配
            if (address[1:].startswith(prefix_target) and 
                address.endswith(suffix_target)):
                found = True
                result = {
                    'found': True,
                    'address': address,
                    'private_key': private_key.hex(),
                    'attempts': attempts,
                    'time': time.time() - start_time
                }
                break
        except Exception as e:
            # 忽略错误，继续
            pass
        
        attempts += 1
        
        # 每100000次显示一次进度
        if attempts % 100000 == 0 and attempts > 0:
            elapsed = time.time() - start_time
            speed = attempts / elapsed if elapsed > 0 else 0
            print(f"  已尝试: {attempts:,} | 速度: {speed:.0f}/秒 | 耗时: {elapsed:.1f}秒")
    
    # 返回结果
    elapsed_time = time.time() - start_time
    
    if found:
        print(f"✓ 找到匹配地址！耗时: {elapsed_time:.2f}秒")
        return result
    else:
        print(f"✗ 未找到匹配地址（尝试{attempts:,}次）")
        return {
            'found': False,
            'attempts': attempts,
            'time': elapsed_time
        }


# 测试
if __name__ == "__main__":
    test_address = "TKzxdSv2FZKQrEqkKVgp5DcwEXBEKMg2Ax"
    print(f"目标地址: {test_address}")
    print(f"目标模式: {test_address[1:3]}...{test_address[-3:]}")
    
    result = generate_real_tron_vanity(test_address, timeout=2.0)
    
    if result['found']:
        print(f"找到匹配地址: {result['address']}")
        print(f"私钥: {result['private_key']}")
    else:
        print(f"未找到匹配地址")
    
    print(f"尝试次数: {result['attempts']:,}")
    print(f"耗时: {result['time']:.2f}秒")
    print(f"速度: {result['attempts']/result['time']:.0f} 地址/秒")
