#!/usr/bin/env python3
"""
最终集成测试 - 测试完整的后缀匹配流程
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.generators.vanity_generator import generate_similar_address


async def test_integration():
    """测试完整的地址生成流程"""
    
    # 测试地址列表
    test_addresses = [
        "TT1LT2H34YMurdmW9Hkxuy2hCbxzekjF5N",  # 真实地址
        "T" + "X" * 28 + "AAAAA",              # 简单后缀
        "T" + "X" * 28 + "12345",              # 数字后缀
    ]
    
    for addr in test_addresses:
        print(f"\n{'='*60}")
        print(f"测试地址: {addr}")
        print(f"目标后缀: {addr[-5:]}")
        print(f"{'='*60}")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            result = await generate_similar_address(
                original_address=addr,
                use_gpu=True,
                timeout=60
            )
            
            elapsed = asyncio.get_event_loop().time() - start_time
            
            if result['success']:
                print(f"✅ 生成成功！")
                print(f"   原始地址: {result['original_address']}")
                print(f"   生成地址: {result['generated_address']}")
                print(f"   私钥: {result['private_key']}")
                print(f"   类型: {result['address_type']}")
                print(f"   耗时: {elapsed:.2f}秒")
                
                # 验证后缀
                original_suffix = result['original_address'][-5:]
                generated_suffix = result['generated_address'][-5:]
                
                if original_suffix == generated_suffix:
                    print(f"   ✅ 后缀匹配成功: {generated_suffix}")
                else:
                    print(f"   ⚠️  后缀不匹配: {original_suffix} != {generated_suffix}")
                    print(f"   可能使用了前缀匹配作为回退方案")
                    
            else:
                print(f"❌ 生成失败: {result.get('error', '未知错误')}")
                
        except Exception as e:
            print(f"❌ 异常: {e}")
            import traceback
            traceback.print_exc()


async def test_api_endpoint():
    """测试API端点"""
    try:
        from app.utils.vanitygen_plusplus import generate_trx_with_vpp
        
        print(f"\n\n{'='*60}")
        print("测试API层级调用")
        print(f"{'='*60}")
        
        test_addr = "TT1LT2H34YMurdmW9Hkxuy2hCbxzekjF5N"
        result = await generate_trx_with_vpp(test_addr)
        
        if result:
            print(f"✅ API调用成功")
            print(f"   地址: {result['address']}")
            print(f"   私钥: {result['private_key']}")
            print(f"   类型: {result['type']}")
            
            if result['address'][-5:] == test_addr[-5:]:
                print(f"   ✅ 后缀匹配: {result['address'][-5:]}")
            else:
                print(f"   ⚠️  使用了前缀匹配")
        else:
            print("❌ API调用失败")
            
    except Exception as e:
        print(f"❌ API测试异常: {e}")


if __name__ == "__main__":
    print("=== TRON地址后缀匹配最终集成测试 ===\n")
    
    # 启用调试
    os.environ["DEBUG"] = "1"
    
    # 运行测试
    asyncio.run(test_integration())
    asyncio.run(test_api_endpoint())
    
    print("\n\n✅ 所有测试完成！")
