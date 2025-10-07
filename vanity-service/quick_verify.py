#!/usr/bin/env python3
"""
快速验证profanity集成是否工作
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.vanitygen_plusplus import generate_trx_with_profanity


async def main():
    # 测试一个简单的后缀
    test_address = "T" + "X" * 28 + "AAAAA"
    
    print(f"测试地址: {test_address}")
    print(f"目标后缀: AAAAA")
    print("生成中...\n")
    
    # 启用调试
    os.environ["DEBUG"] = "1"
    
    result = await generate_trx_with_profanity(test_address)
    
    if result:
        print(f"\n✅ 成功!")
        print(f"地址: {result['address']}")
        print(f"私钥: {result['private_key']}")
        print(f"后缀验证: {result['address'][-5:]} == AAAAA ✓" if result['address'][-5:] == "AAAAA" else f"后缀验证: {result['address'][-5:]} != AAAAA ✗")
    else:
        print("\n❌ 生成失败")


if __name__ == "__main__":
    asyncio.run(main())
