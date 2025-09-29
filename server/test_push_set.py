#!/usr/bin/env python3
"""
测试PUSH_SET功能是否正常工作
"""
import asyncio
import json
import websockets
import sys

async def test_push_set(device_id: str, text: str, ws_url: str = "ws://localhost:8001/ws/clipboard", token: str = "dev-token"):
    """测试向指定设备推送内容"""
    uri = f"{ws_url}?token={token}&deviceId=test-pusher"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"已连接到: {uri}")
            
            # 发送PUSH_SET消息到指定设备
            push_message = {
                "type": "PUSH_SET",
                "targetDeviceId": device_id,  # 目标设备ID
                "set": {
                    "format": "text/plain",
                    "text": text
                },
                "reason": "[测试] 推送测试"
            }
            
            print(f"发送PUSH_SET消息: {json.dumps(push_message, ensure_ascii=False)}")
            await websocket.send(json.dumps(push_message))
            
            # 等待一会儿看是否有响应
            await asyncio.sleep(2)
            
            print("推送完成")
            
    except Exception as e:
        print(f"错误: {e}")

async def direct_push_test(device_id: str, text: str):
    """直接测试推送到已连接的设备"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from main import connected_clients
    
    if device_id in connected_clients:
        ws = connected_clients[device_id]
        push_message = {
            "type": "PUSH_SET",
            "set": {
                "format": "text/plain",
                "text": text
            },
            "reason": "[测试] 直接推送测试"
        }
        
        push_json = json.dumps(push_message)
        print(f"直接发送PUSH_SET到设备 {device_id}: {push_json}")
        await ws.send_text(push_json)
        print("推送完成")
    else:
        print(f"设备 {device_id} 未连接")
        print(f"当前连接的设备: {list(connected_clients.keys())}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python test_push_set.py <device_id> <text>")
        print("示例: python test_push_set.py 66bca5c0-3667-4e62-9da8-6bbfbb374e0d '测试推送内容'")
        sys.exit(1)
    
    device_id = sys.argv[1]
    text = sys.argv[2]
    
    # 运行测试
    asyncio.run(test_push_set(device_id, text))
