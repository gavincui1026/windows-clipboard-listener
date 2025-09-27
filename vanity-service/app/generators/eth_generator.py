"""
ETH/BNB地址生成器（CPU版本）
"""
import time
import asyncio
from eth_account import Account
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Optional


def generate_eth_worker(pattern_info):
    """工作进程：生成ETH地址"""
    prefix_target = pattern_info['prefix'].lower()
    suffix_target = pattern_info['suffix'].lower()
    max_attempts = pattern_info['max_attempts']
    worker_id = pattern_info['worker_id']
    
    attempts = 0
    found = False
    result = None
    
    while attempts < max_attempts and not found:
        # 生成新账户
        account = Account.create()
        address = account.address.lower()
        
        # 检查是否匹配（跳过0x前缀）
        if (address[2:].startswith(prefix_target) and 
            address.endswith(suffix_target)):
            found = True
            result = {
                'found': True,
                'address': account.address,
                'private_key': account.key.hex(),
                'attempts': attempts,
                'worker_id': worker_id
            }
        
        attempts += 1
    
    if not found:
        result = {
            'found': False,
            'attempts': attempts,
            'worker_id': worker_id
        }
    
    return result


async def generate_eth_vanity(prefix: str, suffix: str, timeout: float = 1.5) -> Optional[Dict]:
    """
    生成ETH/BNB虚荣地址
    
    Args:
        prefix: 前缀模式（不包含0x）
        suffix: 后缀模式
        timeout: 超时时间
    
    Returns:
        生成结果字典
    """
    import multiprocessing
    num_workers = multiprocessing.cpu_count()
    
    # 估算难度
    # 十六进制字符，匹配5位
    difficulty = 16 ** (len(prefix) + len(suffix))
    attempts_per_worker = max(10000, difficulty // num_workers // 10)
    
    # 准备工作参数
    worker_params = []
    for i in range(num_workers):
        worker_params.append({
            'prefix': prefix,
            'suffix': suffix,
            'max_attempts': attempts_per_worker,
            'worker_id': i
        })
    
    # 使用进程池
    start_time = time.time()
    total_attempts = 0
    found_result = None
    
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交初始任务
        futures = []
        for params in worker_params:
            future = loop.run_in_executor(executor, generate_eth_worker, params)
            futures.append(future)
        
        # 等待结果
        while time.time() - start_time < timeout and not found_result:
            done, pending = await asyncio.wait(futures, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
            
            for future in done:
                result = await future
                total_attempts += result['attempts']
                
                if result['found']:
                    found_result = result
                    # 取消其他任务
                    for f in pending:
                        f.cancel()
                    break
                else:
                    # 提交新任务
                    if time.time() - start_time < timeout:
                        new_params = {
                            'prefix': prefix,
                            'suffix': suffix,
                            'max_attempts': attempts_per_worker,
                            'worker_id': result['worker_id']
                        }
                        new_future = loop.run_in_executor(executor, generate_eth_worker, new_params)
                        futures.append(new_future)
            
            # 更新待处理任务列表
            futures = list(pending)
    
    # 返回结果
    if found_result:
        return {
            'address': found_result['address'],
            'private_key': found_result['private_key'],
            'type': 'ETH',
            'attempts': total_attempts
        }
    else:
        return None
