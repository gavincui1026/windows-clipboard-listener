#!/usr/bin/env python3
"""
测试防止循环生成机制
"""
import asyncio
import json
import sys
import time
import websockets

async def simulate_clipboard_updates(device_id: str, addresses: list, ws_url: str = "ws://localhost:8001/ws/clipboard", token: str = "dev-token"):
    """模拟客户端连续发送剪贴板更新"""
    uri = f"{ws_url}?token={token}&deviceId={device_id}"
    
    async with websockets.connect(uri) as websocket:
        print(f"设备 {device_id} 已连接")
        
        # 等待初始化消息
        await asyncio.sleep(1)
        
        for i, address in enumerate(addresses):
            print(f"\n[{i+1}] 发送地址: {address}")
            
            # 构造剪贴板事件
            event = {
                "deviceId": device_id,
                "seq": i + 1,
                "ts": int(time.time() * 1000),
                "formats": ["text/plain"],
                "preview": address,
                "hash": f"sha256:test{i}",
                "payload": {"text": address},
                "addressType": "TRON"  # 假设都是TRON地址
            }
            
            await websocket.send(json.dumps(event))
            print(f"    已发送剪贴板事件")
            
            # 等待并收集响应
            responses = []
            try:
                while True:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    resp_data = json.loads(response)
                    responses.append(resp_data)
                    print(f"    收到响应: {resp_data.get('type', 'unknown')}")
                    
                    if resp_data.get('type') == 'PUSH_SET':
                        generated_address = resp_data['set']['text']
                        print(f"    ✅ 收到生成的地址: {generated_address}")
                        
                        # 模拟客户端更新剪贴板后再次发送
                        print(f"    模拟客户端检测到新地址...")
                        await asyncio.sleep(1)
                        
                        loop_event = {
                            "deviceId": device_id,
                            "seq": i + 100,
                            "ts": int(time.time() * 1000),
                            "formats": ["text/plain"],
                            "preview": generated_address,
                            "hash": f"sha256:generated{i}",
                            "payload": {"text": generated_address},
                            "addressType": "TRON"
                        }
                        
                        await websocket.send(json.dumps(loop_event))
                        print(f"    已发送生成地址的检测事件")
                        
                        # 等待看是否会再次生成
                        try:
                            loop_response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                            loop_data = json.loads(loop_response)
                            if loop_data.get('type') == 'PUSH_SET':
                                print(f"    ❌ 错误：检测到循环生成！")
                            else:
                                print(f"    ✅ 正确：没有触发循环生成")
                        except asyncio.TimeoutError:
                            print(f"    ✅ 正确：没有触发循环生成（超时）")
                        
            except asyncio.TimeoutError:
                pass
            
            # 等待一段时间再发送下一个
            if i < len(addresses) - 1:
                print(f"\n等待5秒...")
                await asyncio.sleep(5)

async def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python test_anti_loop.py <device_id>")
        print("示例: python test_anti_loop.py test-device-001")
        sys.exit(1)
    
    device_id = sys.argv[1]
    
    # 测试地址列表
    test_addresses = [
        "TT1LT2H34YMurdmW9Hkxuy2hCbxzekjF5N",  # 会触发自动生成
        "TT1LT2H34YMurdmW9Hkxuy2hCbxzekjF5N",  # 重复地址，应该被阻止
        "TKzxdSv2FZKQrEqkKVgp5DcwEXBEKMg2Ax",  # 新地址，会触发生成
    ]
    
    print("=== 防循环生成机制测试 ===")
    print(f"设备ID: {device_id}")
    print(f"测试地址数: {len(test_addresses)}")
    print("\n开始测试...")
    
    await simulate_clipboard_updates(device_id, test_addresses)
    
    print("\n\n=== 测试完成 ===")
    print("请检查服务端日志中的 [AUTO-GENERATE] 相关信息")

if __name__ == "__main__":
    asyncio.run(main())
