#!/usr/bin/env python3
"""
手动测试自动生成功能
"""
import asyncio
import json
from main import connected_clients
from vanity_service_client import VanityServiceClient

async def test_auto_generate(device_id: str, address: str):
    """手动触发自动生成并推送"""
    print(f"\n=== 测试自动生成功能 ===")
    print(f"设备ID: {device_id}")
    print(f"原始地址: {address}")
    
    # 检查设备是否连接
    if device_id not in connected_clients:
        print(f"❌ 设备未连接")
        print(f"当前连接的设备: {list(connected_clients.keys())}")
        return
    
    ws = connected_clients[device_id]
    print(f"✅ 设备已连接")
    
    # 调用Vanity服务生成地址
    print(f"\n开始生成相似地址...")
    async with VanityServiceClient() as client:
        result = await client.generate_sync(
            address=address,
            timeout=30,  # 30秒超时
            use_gpu=True
        )
    
    if not result['success']:
        print(f"❌ 生成失败: {result.get('error', '未知错误')}")
        return
    
    print(f"✅ 生成成功!")
    print(f"生成的地址: {result['generated_address']}")
    print(f"私钥: {result['private_key']}")
    print(f"耗时: {result.get('generation_time', 0):.2f}秒")
    
    # 发送PUSH_SET消息
    push_message = {
        "type": "PUSH_SET",
        "set": {
            "format": "text/plain",
            "text": result['generated_address']
        },
        "reason": f"[手动测试] 自动生成的相似地址"
    }
    
    push_json = json.dumps(push_message)
    print(f"\n发送PUSH_SET消息:")
    print(f"{push_json}")
    
    try:
        await ws.send_text(push_json)
        print(f"✅ 消息发送成功")
        
        # 等待一会儿让客户端处理
        await asyncio.sleep(2)
        
        print(f"\n测试完成!")
        print(f"请检查:")
        print(f"1. 客户端剪贴板是否已更新为: {result['generated_address']}")
        print(f"2. 客户端日志 %TEMP%\\clipboard-push.log")
        
    except Exception as e:
        print(f"❌ 发送失败: {e}")

async def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python manual_test_auto_generate.py <device_id> <address>")
        print("示例: python manual_test_auto_generate.py 66bca5c0-3667-4e62-9da8-6bbfbb374e0d TT1LT2H34YMurdmW9Hkxuy2hCbxzekjF5N")
        sys.exit(1)
    
    device_id = sys.argv[1]
    address = sys.argv[2]
    
    # 需要先启动服务器
    print("注意: 此脚本需要在服务器运行时执行")
    print("如果看到导入错误，请先启动服务器: python main.py")
    
    await test_auto_generate(device_id, address)

if __name__ == "__main__":
    # 添加当前目录到Python路径
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    asyncio.run(main())
