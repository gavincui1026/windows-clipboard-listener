#!/usr/bin/env python3
"""
测试后缀匹配地址生成
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.vanitygen_plusplus import generate_trx_with_profanity


async def test_suffix_generation():
    """测试后缀匹配生成"""
    # 测试地址（必须是34位）
    test_address = "TT1LT2H34YMurdmW9Hkxuy2hCbxzekjF5N"
    
    print(f"原始地址: {test_address}")
    print(f"地址长度: {len(test_address)}")
    print(f"目标后缀: {test_address[-5:]}")
    print("="*60)
    
    # 测试profanity-tron后缀匹配
    print("\n开始生成...")
    result = await generate_trx_with_profanity(test_address)
    
    if result:
        print(f"\n✅ 生成成功!")
        print(f"生成地址: {result['address']}")
        print(f"私钥: {result['private_key']}")
        print(f"后缀验证: {result['address'][-5:]} == {test_address[-5:]} ✓" 
              if result['address'][-5:] == test_address[-5:] else 
              f"后缀验证: {result['address'][-5:]} != {test_address[-5:]} ✗")
    else:
        print("\n❌ 生成失败")


if __name__ == "__main__":
    print("TRON地址后缀匹配生成测试")
    print("="*60)
    
    # 启用调试
    os.environ["DEBUG"] = "1"
    
    asyncio.run(test_suffix_generation())
