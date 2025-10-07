#!/usr/bin/env python3
"""
测试后缀匹配地址生成
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.vanitygen_plusplus import generate_trx_with_profanity, generate_trx_with_vpp


async def test_suffix_generation():
    """测试后缀匹配生成"""
    # 测试地址
    test_address = "TT1LT2H34YMurdmW9Hkxuy2hCbxzekjF5N"
    
    print(f"原始地址: {test_address}")
    print(f"目标后缀: {test_address[-5:]}")
    print("="*60)
    
    # 测试profanity-tron后缀匹配
    print("\n测试profanity-tron后缀匹配...")
    result = await generate_trx_with_profanity(test_address)
    
    if result:
        print(f"✅ 生成成功!")
        print(f"   地址: {result['address']}")
        print(f"   私钥: {result['private_key']}")
        print(f"   后缀匹配: {result['address'][-5:]} == {test_address[-5:]}")
    else:
        print("❌ profanity-tron生成失败")
    
    # 测试完整的generate_trx_with_vpp函数
    print("\n\n测试完整的generate_trx_with_vpp函数...")
    result2 = await generate_trx_with_vpp(test_address)
    
    if result2:
        print(f"✅ 生成成功!")
        print(f"   地址: {result2['address']}")
        print(f"   私钥: {result2['private_key']}")
        if result2['address'][-5:] == test_address[-5:]:
            print(f"   ✅ 后缀匹配: {result2['address'][-5:]}")
        else:
            print(f"   ⚠️  前缀匹配: {result2['address'][:5]}")
    else:
        print("❌ 生成失败")


async def test_custom_suffix():
    """测试自定义后缀"""
    # 创建一个容易生成的测试地址（后缀为AAAAA）
    test_address = "T" + "X" * 28 + "AAAAA"
    
    print(f"\n\n测试自定义后缀...")
    print(f"测试地址: {test_address}")
    print(f"目标后缀: AAAAA")
    
    result = await generate_trx_with_profanity(test_address)
    
    if result:
        print(f"✅ 生成成功!")
        print(f"   地址: {result['address']}")
        print(f"   私钥: {result['private_key']}")
        print(f"   验证: 后缀为 {result['address'][-5:]}")
    else:
        print("❌ 生成失败")


if __name__ == "__main__":
    print("TRON地址后缀匹配生成测试")
    print("="*60)
    
    asyncio.run(test_suffix_generation())
    asyncio.run(test_custom_suffix())
