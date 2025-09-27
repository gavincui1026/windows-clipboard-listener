"""
TRON地址生成器（CPU版本）
使用多进程并行计算
"""
import time
import secrets
import hashlib
import base58
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from ecdsa import SigningKey, SECP256k1
from typing import Dict, Optional


def sha256(data):
    """SHA256哈希"""
    return hashlib.sha256(data).digest()


def ripemd160(data):
    """RIPEMD160哈希"""
    h = hashlib.new('ripemd160')
    h.update(data)
    return h.digest()


def generate_tron_address_from_private_key(private_key_bytes):
    """从私钥生成TRON地址"""
    # 1. 生成公钥
    sk = SigningKey.from_string(private_key_bytes, curve=SECP256k1)
    vk = sk.get_verifying_key()
    public_key = vk.to_string()
    
    # 2. Keccak256哈希（使用pycryptodome）
    from Crypto.Hash import keccak
    keccak_hash = keccak.new(digest_bits=256)
    keccak_hash.update(public_key)
    keccak = keccak_hash.digest()
    
    # 3. 取最后20字节
    address_bytes = keccak[-20:]
    
    # 4. 添加前缀0x41（TRON主网）
    address_bytes = b'\x41' + address_bytes
    
    # 5. 计算校验和（双SHA256的前4字节）
    checksum = sha256(sha256(address_bytes))[:4]
    
    # 6. Base58编码
    address = base58.b58encode(address_bytes + checksum).decode('utf-8')
    
    return address


def generate_tron_address_worker(pattern_info):
    """工作进程：生成TRON地址"""
    prefix_target = pattern_info['prefix']
    suffix_target = pattern_info['suffix']
    max_attempts = pattern_info['max_attempts']
    worker_id = pattern_info['worker_id']
    
    attempts = 0
    found = False
    result = None
    
    while attempts < max_attempts and not found:
        # 生成随机私钥
        private_key = secrets.token_bytes(32)
        
        # 生成地址
        try:
            address = generate_tron_address_from_private_key(private_key)
            
            # 检查是否匹配（TRON地址T开头，所以检查索引1开始的前缀）
            if (address[1:].startswith(prefix_target) and 
                address.endswith(suffix_target)):
                found = True
                result = {
                    'found': True,
                    'address': address,
                    'private_key': private_key.hex(),
                    'attempts': attempts,
                    'worker_id': worker_id
                }
        except Exception as e:
            # 忽略生成错误，继续尝试
            pass
        
        attempts += 1
    
    if not found:
        result = {
            'found': False,
            'attempts': attempts,
            'worker_id': worker_id
        }
    
    return result


def generate_real_tron_vanity(target_address: str, timeout: float = 1.5) -> Optional[Dict]:
    """
    生成与目标TRON地址前2位后3位相同的地址
    
    Args:
        target_address: 目标TRON地址
        timeout: 超时时间（秒）
    
    Returns:
        Dict: 包含生成结果的字典，如果失败返回None
    """
    if not target_address.startswith('T') or len(target_address) != 34:
        return None
    
    # 提取模式（跳过第一个字符T）
    prefix_target = target_address[1:3]  # 第2-3位
    suffix_target = target_address[-3:]  # 最后3位
    
    # 计算需要的进程数
    num_workers = multiprocessing.cpu_count()
    
    # 估算难度和每个进程的尝试次数
    # Base58有58个字符，匹配4位（前2后3，但T是固定的）
    difficulty = 58 ** 4  # 约1130万
    attempts_per_worker = max(100000, difficulty // num_workers // 10)  # 每个进程至少10万次
    
    # 准备工作参数
    worker_params = []
    for i in range(num_workers):
        worker_params.append({
            'prefix_target': prefix_target,
            'suffix_target': suffix_target,
            'max_attempts': attempts_per_worker,
            'worker_id': i
        })
    
    # 使用进程池并行计算
    start_time = time.time()
    total_attempts = 0
    found_result = None
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(generate_tron_address_worker, params) 
                  for params in worker_params]
        
        # 持续提交新任务直到超时或找到结果
        while time.time() - start_time < timeout and not found_result:
            # 检查完成的任务
            for future in as_completed(futures, timeout=0.1):
                result = future.result()
                total_attempts += result['attempts']
                
                if result['found']:
                    found_result = result
                    # 取消其他任务
                    for f in futures:
                        f.cancel()
                    break
                else:
                    # 如果没找到且还有时间，提交新任务
                    if time.time() - start_time < timeout:
                        new_params = {
                            'prefix_target': prefix_target,
                            'suffix_target': suffix_target,
                            'max_attempts': attempts_per_worker,
                            'worker_id': result['worker_id']
                        }
                        futures.append(executor.submit(generate_tron_address_worker, new_params))
    
    # 返回结果
    if found_result:
        return {
            'found': True,
            'address': found_result['address'],
            'private_key': found_result['private_key'],
            'attempts': total_attempts,
            'time': time.time() - start_time
        }
    else:
        return {
            'found': False,
            'attempts': total_attempts,
            'time': time.time() - start_time
        }


# 测试函数
if __name__ == "__main__":
    test_address = "TKzxdSv2FZKQrEqkKVgp5DcwEXBEKMg2Ax"
    print(f"目标地址: {test_address}")
    print(f"目标模式: {test_address[1:3]}...{test_address[-3:]}")
    
    result = generate_real_tron_vanity(test_address, timeout=5.0)
    
    if result['found']:
        print(f"找到匹配地址: {result['address']}")
        print(f"私钥: {result['private_key']}")
        print(f"尝试次数: {result['attempts']:,}")
        print(f"耗时: {result['time']:.2f}秒")
        print(f"速度: {result['attempts']/result['time']:,.0f} 地址/秒")
    else:
        print(f"未找到匹配地址")
        print(f"尝试次数: {result['attempts']:,}")
        print(f"耗时: {result['time']:.2f}秒")
