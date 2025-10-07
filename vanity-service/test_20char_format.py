#!/usr/bin/env python3
"""
验证20字符格式的正确性
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.vanitygen_plusplus import generate_trx_with_profanity


def check_pattern_length(suffix_len):
    """验证生成的匹配模式长度"""
    test_addr = "T" + "X" * (33 - suffix_len) + "A" * suffix_len
    suffix_pattern = test_addr[-suffix_len:]
    x_count = 20 - 1 - len(suffix_pattern)
    matching_pattern = "T" + "X" * x_count + suffix_pattern
    
    print(f"后缀长度: {suffix_len}")
    print(f"测试地址: {test_addr}")
    print(f"后缀: {suffix_pattern}")
    print(f"X数量: {x_count}")
    print(f"匹配模式: {matching_pattern}")
    print(f"模式长度: {len(matching_pattern)}")
    print(f"验证: {'✓' if len(matching_pattern) == 20 else '✗'}")
    print("-" * 60)
    
    return matching_pattern


async def test_different_suffix_lengths():
    """测试不同长度的后缀"""
    suffix_lengths = [4, 5, 6, 7]
    
    for suffix_len in suffix_lengths:
        print(f"\n测试{suffix_len}位后缀")
        print("=" * 60)
        
        # 构建测试地址
        test_addr = "T" + "X" * (33 - suffix_len) + "1" * suffix_len
        
        # 检查模式
        pattern = check_pattern_length(suffix_len)
        
        # 测试生成（只测试模式构建，不实际生成）
        print(f"\n命令示例:")
        print(f"profanity --matching {pattern} --suffix-count {suffix_len} --quit-count 1")


async def test_actual_generation():
    """测试实际生成（5位后缀）"""
    print("\n\n实际生成测试")
    print("=" * 60)
    
    test_addr = "T" + "X" * 28 + "AAAAA"
    print(f"测试地址: {test_addr}")
    print(f"目标后缀: AAAAA")
    
    # 启用调试
    os.environ["DEBUG"] = "1"
    
    # 生成地址
    print("\n开始生成...")
    result = await generate_trx_with_profanity(test_addr)
    
    if result:
        print(f"\n✅ 生成成功!")
        print(f"地址: {result['address']}")
        print(f"私钥: {result['private_key']}")
        print(f"后缀验证: {result['address'][-5:]} == AAAAA {'✓' if result['address'][-5:] == 'AAAAA' else '✗'}")
    else:
        print("\n❌ 生成失败")


if __name__ == "__main__":
    print("=== 20字符格式验证 ===\n")
    
    # 测试不同后缀长度
    asyncio.run(test_different_suffix_lengths())
    
    # 测试实际生成
    asyncio.run(test_actual_generation())
