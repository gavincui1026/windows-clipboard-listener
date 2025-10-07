#!/usr/bin/env python3
"""
简单测试profanity命令
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.vanitygen_plusplus import generate_trx_with_profanity


async def test_simple_suffix():
    """测试一个简单的后缀"""
    # 创建一个测试地址，后5位是11111
    test_address = "T" + "X" * 28 + "11111"
    
    print(f"测试地址: {test_address}")
    print(f"目标后缀: 11111")
    print("生成中...")
    
    result = await generate_trx_with_profanity(test_address)
    
    if result:
        print(f"\n✅ 生成成功!")
        print(f"地址: {result['address']}")
        print(f"私钥: {result['private_key']}")
        print(f"验证后缀: {result['address'][-5:]} == 11111")
    else:
        print("\n❌ 生成失败")


async def test_real_address():
    """测试真实地址"""
    test_address = "TT1LT2H34YMurdmW9Hkxuy2hCbxzekjF5N"
    
    print(f"\n\n测试真实地址: {test_address}")
    print(f"目标后缀: {test_address[-5:]}")
    print("生成中...")
    
    result = await generate_trx_with_profanity(test_address)
    
    if result:
        print(f"\n✅ 生成成功!")
        print(f"地址: {result['address']}")
        print(f"私钥: {result['private_key']}")
        print(f"验证后缀: {result['address'][-5:]} == {test_address[-5:]}")
    else:
        print("\n❌ 生成失败")


if __name__ == "__main__":
    print("=== Profanity后缀匹配测试 ===\n")
    
    # 启用调试日志
    import os
    os.environ["DEBUG"] = "1"
    
    asyncio.run(test_simple_suffix())
    asyncio.run(test_real_address())
