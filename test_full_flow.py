"""
测试完整的地址生成流程
"""
import asyncio
import sys
import os

# 添加server目录到path
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from vanity_service_client import VanityServiceClient

async def test_full_flow():
    """测试完整流程"""
    print("=== 测试完整地址生成流程 ===\n")
    
    # 测试地址
    test_addresses = [
        "TT1LT2H34YMurdmW9Hkxuy2hCbxzekjF5N",  # TRON
        "0x742d35Cc6634C0532925a3b844Bc9e7595f6E321",  # ETH
    ]
    
    async with VanityServiceClient() as client:
        # 1. 健康检查
        print("[1] 健康检查...")
        if await client.health_check():
            print("✅ Vanity服务正常运行\n")
        else:
            print("❌ Vanity服务不可用")
            return
        
        # 2. 测试地址生成
        for address in test_addresses:
            print(f"[2] 测试生成地址: {address}")
            
            result = await client.generate_sync(
                address=address,
                timeout=1.5,
                use_gpu=True
            )
            
            print(f"返回结果: {result}")
            
            if result.get('success'):
                print(f"✅ 生成成功!")
                print(f"   原地址: {result.get('original_address')}")
                print(f"   新地址: {result.get('generated_address')}")
                print(f"   私钥: {result.get('private_key', '')[:32]}...")
                print(f"   类型: {result.get('address_type')}")
                print(f"   耗时: {result.get('generation_time', 0):.3f}秒")
            else:
                print(f"❌ 生成失败: {result.get('error')}")
            
            print()

if __name__ == "__main__":
    asyncio.run(test_full_flow())
