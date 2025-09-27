"""Telegram Bot处理回复消息并替换剪贴板"""
import asyncio
import aiohttp
import json
from typing import Optional
from db import get_session, SysSettings, MessageDeviceMapping, Device
from fastapi import WebSocket


class TelegramBot:
    def __init__(self, bot_token: str):
        self.bot_token = bot_token
        self.update_offset = 0
        self.running = False
        
    async def get_updates(self) -> list:
        """获取Telegram更新"""
        url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
        params = {
            "offset": self.update_offset,
            "timeout": 30,  # 长轮询
            "allowed_updates": ["message"]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        updates = data.get("result", [])
                        
                        # 更新偏移量
                        if updates:
                            self.update_offset = updates[-1]["update_id"] + 1
                            
                        return updates
                    else:
                        print(f"获取Telegram更新失败: {resp.status}")
                        return []
        except Exception as e:
            print(f"获取Telegram更新出错: {e}")
            return []
    
    async def process_updates(self, updates: list, connected_clients: dict):
        """处理Telegram更新"""
        for update in updates:
            message = update.get("message", {})
            
            # 检查是否是回复消息
            reply_to = message.get("reply_to_message")
            if not reply_to:
                continue
                
            # 获取被回复消息的ID
            reply_to_id = reply_to.get("message_id")
            if not reply_to_id:
                continue
                
            # 获取新内容
            new_text = message.get("text", "").strip()
            if not new_text:
                continue
                
            # 查找对应的设备
            with get_session() as db:
                mapping = db.query(MessageDeviceMapping).filter(
                    MessageDeviceMapping.message_id == reply_to_id
                ).first()
                
                if not mapping:
                    # 回复的消息不是地址通知或已过期
                    chat_id = message.get("chat", {}).get("id")
                    message_id = message.get("message_id")
                    if chat_id and message_id:
                        await self.send_reply(chat_id, message_id, "❌ 无效操作，请回复地址通知消息")
                    continue
                    
                device_id = mapping.device_id
                
                # 检查设备是否有地址内容
                device = db.query(Device).filter(Device.device_id == device_id).first()
                if not device or not device.last_clip_text:
                    print(f"设备 {device_id} 剪贴板为空，无法推送")
                    # 发送失败回复
                    chat_id = message.get("chat", {}).get("id")
                    message_id = message.get("message_id")
                    if chat_id and message_id:
                        await self.send_reply(chat_id, message_id, "❌ 剪切板暂不可替换")
                    continue
            
            # 推送到设备
            ws = connected_clients.get(device_id)
            chat_id = message.get("chat", {}).get("id")
            message_id = message.get("message_id")
            
            # 准备回复
            reply_text = ""
            success = False
            
            if ws:
                try:
                    msg = {
                        "type": "PUSH_SET",
                        "set": {
                            "format": "text/plain",
                            "text": new_text
                        },
                        "reason": "telegram-reply"
                    }
                    await ws.send_text(json.dumps(msg))
                    print(f"已通过Telegram回复推送内容到设备 {device_id}: {new_text}")
                    reply_text = "✅ 替换成功"
                    success = True
                except Exception as e:
                    print(f"推送失败: {e}")
                    reply_text = "❌ 剪切板暂不可替换"
            else:
                print(f"设备 {device_id} 离线，无法推送")
                reply_text = "❌ 剪切板暂不可替换"
            
            # 发送回复到Telegram
            if chat_id and message_id:
                await self.send_reply(chat_id, message_id, reply_text)
    
    async def send_reply(self, chat_id: int, reply_to_message_id: int, text: str):
        """发送回复消息到Telegram"""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "reply_to_message_id": reply_to_message_id
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        print(f"已发送回复: {text}")
                    else:
                        error = await resp.text()
                        print(f"发送回复失败: {error}")
        except Exception as e:
            print(f"发送回复时出错: {e}")
    
    async def run(self, connected_clients: dict):
        """运行机器人"""
        self.running = True
        print(f"Telegram Bot已启动，正在监听消息...")
        
        while self.running:
            try:
                updates = await self.get_updates()
                if updates:
                    await self.process_updates(updates, connected_clients)
            except Exception as e:
                print(f"Telegram Bot运行出错: {e}")
                await asyncio.sleep(5)  # 出错后等待5秒再试
                
    def stop(self):
        """停止机器人"""
        self.running = False


# 全局机器人实例
telegram_bot: Optional[TelegramBot] = None


async def start_telegram_bot(connected_clients: dict):
    """启动Telegram机器人"""
    global telegram_bot
    
    # 获取配置
    with get_session() as db:
        bot_token = db.query(SysSettings).filter(SysSettings.key == "tg_bot_token").first()
        
        if not bot_token or not bot_token.value:
            print("Telegram Bot Token未配置，跳过启动")
            return
    
    # 创建并运行机器人
    telegram_bot = TelegramBot(bot_token.value)
    await telegram_bot.run(connected_clients)


def stop_telegram_bot():
    """停止Telegram机器人"""
    global telegram_bot
    if telegram_bot:
        telegram_bot.stop()
        telegram_bot = None
