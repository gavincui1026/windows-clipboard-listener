"""
统一的地址生成器（vanitygen-plusplus 版）
仅通过 vanitygen-plusplus 生成地址/私钥
"""
import time
from typing import Dict, Optional, Tuple
import asyncio
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from app.utils.vanitygen_plusplus import (
    generate_btc_with_vpp,
    generate_trx_with_vpp,
    generate_eth_with_vpp,
    is_vpp_available,
)

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
    
    # Solana - 不再支持
    # if len(address) in range(32, 45) and not address.startswith(('0x', 'T', '1', '3', 'bc1')):
    #     # 简单检查是否为Base58字符
    #     import base58
    #     try:
    #         base58.b58decode(address)
    #         return 'Solana'
    #     except:
    #         pass
    
    return None


def get_pattern_from_address(address: str, address_type: str) -> Tuple[str, str]:
    """从地址提取前5位作为前缀（不包含固定前缀）"""
    if address_type == 'TRON':
        # T开头，提取T后的前4位（总共匹配前5位）
        prefix = address[1:5] if len(address) > 4 else address[1:]
        suffix = ""
        return prefix, suffix
    
    elif address_type in ['BTC_P2PKH', 'BTC_P2SH', 'BTC_Bech32']:
        # Bitcoin地址，根据类型处理
        if address_type == 'BTC_P2PKH':  # 1开头，提取1后的前4位
            prefix = address[1:5] if len(address) > 4 else address[1:]
        elif address_type == 'BTC_P2SH':  # 3开头，提取3后的前4位
            prefix = address[1:5] if len(address) > 4 else address[1:]
        else:  # bc1开头，提取bc1后的前2位（总共匹配前5位）
            prefix = address[3:5] if len(address) > 4 else address[3:]
        suffix = ""
        return prefix, suffix
    
    elif address_type in ['ETH', 'BNB']:
        # 0x开头，提取0x后的前3位（总共匹配前5位）
        prefix = address[2:5] if len(address) > 4 else address[2:]
        suffix = ""
        return prefix, suffix
    
    return "", ""


async def generate_similar_address(
    original_address: str,
    use_gpu: bool = True,
    timeout: float = 0
) -> Dict:
    """
    生成相似地址的主函数（仅支持 BTC）
    """
    start_time = time.time()
    
    address_type = detect_address_type(original_address)
    
    if not address_type or address_type == 'Unknown':
        return {
            "success": False,
            "error": "剪贴板内容不是支持的加密货币地址"
        }
    
    # 使用 vanitygen-plusplus 生成（BTC / TRX / ETH）
    try:
        if address_type.startswith('BTC'):
            result = await generate_btc_with_vpp(original_address, address_type)
        elif address_type == 'TRON':
            result = await generate_trx_with_vpp(original_address)
        elif address_type == 'ETH':
            result = await generate_eth_with_vpp(original_address)
        else:
            result = None
        if result:
            generation_time = time.time() - start_time
            return {
                "success": True,
                "original_address": original_address,
                "generated_address": result['address'],
                "private_key": result['private_key'],
                "address_type": result.get('type', address_type),
                "balance": "0",
                "attempts": 0,
                "generation_time": generation_time
            }
        return {
            "success": False,
            "error": "vanitygen-plusplus 未找到或未能生成匹配地址"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"生成失败: {e}"
        }


def estimate_difficulty(address_type: str, pattern_length: int = 5) -> int:
    """估算生成难度"""
    if address_type == 'TRON':
        # 调整为匹配后5位（T固定）
        return 58 ** 5
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
