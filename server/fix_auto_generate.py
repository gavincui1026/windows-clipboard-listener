"""
修复现有设备的 auto_generate 字段为默认值 True
"""
from db import get_session, Device

def fix_auto_generate():
    with get_session() as db:
        # 更新所有 auto_generate 为 NULL 的设备
        devices = db.query(Device).filter(Device.auto_generate == None).all()
        
        if devices:
            print(f"找到 {len(devices)} 个设备需要更新 auto_generate 字段")
            for device in devices:
                device.auto_generate = True
                print(f"更新设备 {device.device_id} 的 auto_generate 为 True")
            
            db.commit()
            print("更新完成！")
        else:
            print("没有需要更新的设备")

if __name__ == "__main__":
    fix_auto_generate()
