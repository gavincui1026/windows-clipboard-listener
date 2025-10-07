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
    
    # 构建匹配模式
    suffix = test_address[-5:]
    matching_pattern = "T" + "X" * 15 + suffix
    
    # 显示将要执行的命令
    cmd = f"profanity --matching {matching_pattern} --suffix-count 5 --quit-count 1"
    print(f"\n匹配模式: {matching_pattern} (长度: {len(matching_pattern)})")
    print(f"执行命令: {cmd}")
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


async def test_invalid_address():
    """测试无效地址"""
    print("\n\n测试无效地址（长度不对）")
    print("="*60)
    
    # 测试33位地址（少一位）
    test_address = "TT1LT2H34YMurdmW9Hkxuy2hCbxzekjF5"
    print(f"测试地址: {test_address}")
    print(f"地址长度: {len(test_address)} (应该是34)")
    
    result = await generate_trx_with_profanity(test_address)
    if not result:
        print("\n✅ 正确拒绝了无效地址")
    else:
        print("\n❌ 错误：不应该接受无效地址")


async def test_with_simple_suffix():
    """测试简单后缀（容易生成）"""
    print("\n\n测试简单后缀（11111）")
    print("="*60)
    
    # 使用简单的后缀11111，更容易生成
    test_address = "TT1LT2H34YMurdmW9Hkxuy2hCbxze11111"
    print(f"测试地址: {test_address}")
    print(f"目标后缀: {test_address[-5:]}")
    
    # 构建匹配模式
    suffix = test_address[-5:]
    matching_pattern = "T" + "X" * 15 + suffix
    
    # 显示将要执行的命令
    cmd = f"profanity --matching {matching_pattern} --suffix-count 5 --quit-count 1"
    print(f"\n匹配模式: {matching_pattern} (长度: {len(matching_pattern)})")
    print(f"执行命令: {cmd}")
    
    print("\n开始生成...")
    result = await generate_trx_with_profanity(test_address)
    
    if result:
        print(f"\n生成结果汇总:")
        print(f"- 原始地址: {test_address}")
        print(f"- 生成地址: {result['address']}")
        print(f"- 后缀对比: {test_address[-5:]} -> {result['address'][-5:]}")


if __name__ == "__main__":
    print("TRON地址后缀匹配生成测试")
    print("="*60)
    
    # 启用调试
    os.environ["DEBUG"] = "1"
    
    # 运行各种测试
    asyncio.run(test_suffix_generation())
    asyncio.run(test_invalid_address())
    asyncio.run(test_with_simple_suffix())
