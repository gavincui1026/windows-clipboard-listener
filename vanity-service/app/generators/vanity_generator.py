"""
统一的地址生成器
协调CPU和GPU生成
"""
import time
from typing import Dict, Optional, Tuple
import asyncio
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# 地址类型检测
def detect_address_type(address: str) -> Optional[str]:
    """检测地址类型"""
    if not address:
        return None
    
    # TRON
    if address.startswith('T') and len(address) in [34]:
        return 'TRON'
    
    # Bitcoin
    if address.startswith('1') and len(address) in range(26, 35):
        return 'BTC_P2PKH'
    elif address.startswith('3') and len(address) in range(26, 35):
        return 'BTC_P2SH'
    elif address.startswith('bc1') and len(address) in range(39, 63):
        return 'BTC_Bech32'
    
    # Ethereum/BNB
    if address.startswith('0x') and len(address) == 42:
        return 'ETH'  # 也可能是BNB
    
    # Solana
    if len(address) in range(32, 45) and not address.startswith(('0x', 'T', '1', '3', 'bc1')):
        # 简单检查是否为Base58字符
        import base58
        try:
            base58.b58decode(address)
            return 'Solana'
        except:
            pass
    
    return None


def get_pattern_from_address(address: str, address_type: str) -> Tuple[str, str]:
    """从地址提取前2后3模式"""
    if address_type == 'TRON':
        # TRON地址T开头，提取第2-3位和最后3位
        prefix = address[1:3] if len(address) > 2 else ""
        suffix = address[-3:] if len(address) >= 3 else ""
        return prefix, suffix
    
    elif address_type in ['BTC_P2PKH', 'BTC_P2SH', 'BTC_Bech32']:
        # Bitcoin地址，根据类型处理
        if address_type == 'BTC_P2PKH':  # 1开头
            prefix = address[1:3] if len(address) > 2 else ""
        elif address_type == 'BTC_P2SH':  # 3开头
            prefix = address[1:3] if len(address) > 2 else ""
        else:  # bc1开头
            prefix = address[3:5] if len(address) > 4 else ""
        suffix = address[-3:] if len(address) >= 3 else ""
        return prefix, suffix
    
    elif address_type in ['ETH', 'BNB']:
        # 0x开头，提取0x后的2位和最后3位
        prefix = address[2:4] if len(address) > 3 else ""
        suffix = address[-3:] if len(address) >= 3 else ""
        return prefix, suffix
    
    elif address_type == 'Solana':
        # Solana地址，提取前2位和后3位
        prefix = address[:2]
        suffix = address[-3:] if len(address) >= 3 else ""
        return prefix, suffix
    
    return "", ""


async def generate_similar_address(
    original_address: str,
    use_gpu: bool = True,
    timeout: float = 0  # 无限制
) -> Dict:
    """
    生成相似地址的主函数
    """
    start_time = time.time()
    address_type = detect_address_type(original_address)
    
    if not address_type or address_type == 'Unknown':
        return {
            "success": False,
            "error": "剪贴板内容不是支持的加密货币地址"
        }
    
    # 提取模式
    prefix, suffix = get_pattern_from_address(original_address, address_type)
    if not prefix and not suffix:
        return {
            "success": False,
            "error": "无法提取地址模式"
        }
    
    # 根据地址类型选择生成策略
    generated_address_info = None
    
    # 只使用GPU生成（完全删除CPU逻辑）
    if use_gpu:
        try:
            # 直接使用PyTorch GPU
            import torch
            if torch.cuda.is_available():
                print(f"使用PyTorch GPU: {torch.cuda.get_device_name(0)}")
                from .gpu_torch import generate_address_torch_gpu
                # 使用Ultra GPU模式，2秒超时
                generated_address_info = await generate_address_torch_gpu(original_address, address_type, timeout or 2.0)
            else:
                return {
                    "success": False,
                    "error": "GPU不可用，请检查PyTorch GPU版本是否正确安装"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"GPU生成失败: {e}"
            }
    else:
        return {
            "success": False,
            "error": "该服务仅支持GPU模式，请设置use_gpu=true"
        }
    
    # 返回结果
    if generated_address_info:
        generation_time = time.time() - start_time
        return {
            "success": True,
            "original_address": original_address,
            "generated_address": generated_address_info['address'],
            "private_key": generated_address_info['private_key'],
            "address_type": generated_address_info.get('type', address_type),
            "balance": "0",
            "attempts": generated_address_info.get('attempts', 0),
            "generation_time": generation_time
        }
    else:
        return {
            "success": False,
            "error": f"在{timeout}秒内未能生成匹配的{address_type}地址"
        }


def estimate_difficulty(address_type: str, pattern_length: int = 5) -> int:
    """估算生成难度"""
    if address_type == 'TRON':
        # Base58字符集大小为58，匹配4位（T是固定的）
        return 58 ** 4
    elif address_type in ['ETH', 'BNB']:
        # 十六进制，匹配5位
        return 16 ** 5
    elif address_type.startswith('BTC'):
        # Base58，根据类型不同
        if address_type == 'BTC_Bech32':
            return 32 ** 5  # Bech32字符集
        else:
            return 58 ** 4
    else:
        return 58 ** 5  # 默认估算
