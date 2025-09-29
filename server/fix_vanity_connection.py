#!/usr/bin/env python3
"""
修复VPS服务器无法连接vanity服务的问题
"""
import os
import sys
import json
import aiohttp
import asyncio
import subprocess
from typing import Dict, Tuple, Optional

class VanityConnectionFixer:
    """修复Vanity服务连接问题"""
    
    def __init__(self):
        self.current_url = os.getenv("VANITY_SERVICE_URL") or "https://trainers-pads-switches-links.trycloudflare.com"
        
    async def test_connection(self, url: str) -> Tuple[bool, str]:
        """测试连接"""
        try:
            print(f"测试连接: {url}")
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if 200 <= resp.status < 300:
                        return True, "连接成功"
                    else:
                        return False, f"HTTP状态码: {resp.status}"
        except aiohttp.ClientError as e:
            return False, f"连接错误: {str(e)}"
        except Exception as e:
            return False, f"未知错误: {str(e)}"
    
    def check_dns(self, domain: str) -> bool:
        """检查DNS解析"""
        try:
            result = subprocess.run(
                ["nslookup", domain],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def check_network_route(self, domain: str) -> bool:
        """检查网络路由"""
        try:
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "2", domain],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    async def diagnose(self) -> Dict[str, any]:
        """诊断连接问题"""
        results = {
            "current_url": self.current_url,
            "issues": [],
            "solutions": []
        }
        
        # 1. 测试当前URL连接
        connected, msg = await self.test_connection(self.current_url)
        if not connected:
            results["issues"].append(f"无法连接到当前URL: {msg}")
            
            # 2. 检查是否是Cloudflare域名
            if "trycloudflare.com" in self.current_url:
                results["issues"].append("使用的是Cloudflare Tunnel域名，可能被VPS网络限制")
                results["solutions"].append("使用本地部署的vanity服务")
                results["solutions"].append("使用直接的IP地址或域名")
                
            # 3. 检查DNS
            domain = self.current_url.split("://")[1].split("/")[0].split(":")[0]
            if not self.check_dns(domain):
                results["issues"].append(f"DNS解析失败: {domain}")
                results["solutions"].append("检查VPS的DNS设置")
                results["solutions"].append("尝试使用公共DNS (8.8.8.8, 1.1.1.1)")
                
            # 4. 检查网络路由
            if not self.check_network_route(domain):
                results["issues"].append(f"无法ping通域名: {domain}")
                results["solutions"].append("检查VPS防火墙设置")
                results["solutions"].append("检查VPS网络配置")
        
        # 5. 测试本地vanity服务
        local_url = "http://localhost:8002"
        local_connected, local_msg = await self.test_connection(local_url)
        if local_connected:
            results["solutions"].append(f"发现本地vanity服务运行在: {local_url}")
        
        return results
    
    def create_env_file(self) -> str:
        """创建或更新.env文件"""
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        env_content = []
        
        # 读取现有配置
        if os.path.exists(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip().startswith('VANITY_SERVICE_URL'):
                        env_content.append(line.rstrip())
        
        # 添加新的VANITY_SERVICE_URL
        env_content.append('\n# Vanity服务配置 (使用本地服务避免网络问题)')
        env_content.append('VANITY_SERVICE_URL=http://localhost:8002')
        
        # 写入文件
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(env_content))
        
        return env_path
    
    async def apply_fixes(self) -> Dict[str, any]:
        """应用修复方案"""
        results = {
            "actions": [],
            "recommendations": []
        }
        
        # 1. 诊断问题
        diagnosis = await self.diagnose()
        results["diagnosis"] = diagnosis
        
        # 2. 如果是Cloudflare域名问题，建议使用本地服务
        if any("trycloudflare.com" in issue for issue in diagnosis.get("issues", [])):
            results["recommendations"].append({
                "priority": "HIGH",
                "action": "部署本地vanity服务",
                "command": "cd /path/to/project && docker-compose up -d vanity-service"
            })
            
            # 创建.env文件
            env_path = self.create_env_file()
            results["actions"].append(f"创建了.env文件: {env_path}")
            results["actions"].append("设置VANITY_SERVICE_URL=http://localhost:8002")
            
        # 3. 如果是DNS问题
        if any("DNS" in issue for issue in diagnosis.get("issues", [])):
            results["recommendations"].append({
                "priority": "MEDIUM",
                "action": "修改DNS设置",
                "commands": [
                    "echo 'nameserver 8.8.8.8' >> /etc/resolv.conf",
                    "echo 'nameserver 1.1.1.1' >> /etc/resolv.conf"
                ]
            })
        
        # 4. 提供直接IP连接选项
        results["recommendations"].append({
            "priority": "LOW",
            "action": "如果vanity服务部署在其他服务器，使用直接IP",
            "example": "VANITY_SERVICE_URL=http://YOUR_SERVER_IP:8002"
        })
        
        return results

async def main():
    """主函数"""
    print("=== Vanity服务连接问题诊断工具 ===\n")
    
    fixer = VanityConnectionFixer()
    
    # 诊断问题
    print("正在诊断连接问题...")
    diagnosis = await fixer.diagnose()
    
    print("\n诊断结果:")
    print(f"当前配置URL: {diagnosis['current_url']}")
    
    if diagnosis['issues']:
        print("\n发现的问题:")
        for issue in diagnosis['issues']:
            print(f"  - {issue}")
    
    if diagnosis['solutions']:
        print("\n建议的解决方案:")
        for i, solution in enumerate(diagnosis['solutions'], 1):
            print(f"  {i}. {solution}")
    
    # 询问是否应用修复
    print("\n是否应用自动修复方案? (y/n): ", end='')
    response = input().strip().lower()
    
    if response == 'y':
        print("\n正在应用修复...")
        fix_results = await fixer.apply_fixes()
        
        if fix_results['actions']:
            print("\n已执行的操作:")
            for action in fix_results['actions']:
                print(f"  ✓ {action}")
        
        if fix_results['recommendations']:
            print("\n推荐的后续操作:")
            for rec in fix_results['recommendations']:
                print(f"\n[{rec['priority']}] {rec['action']}")
                if 'command' in rec:
                    print(f"  命令: {rec['command']}")
                elif 'commands' in rec:
                    print("  命令:")
                    for cmd in rec['commands']:
                        print(f"    {cmd}")
                if 'example' in rec:
                    print(f"  示例: {rec['example']}")
        
        print("\n修复完成！请重启服务以应用更改:")
        print("  systemctl restart your-api-service")
        print("  或")
        print("  docker-compose restart api")
    else:
        print("\n取消修复。")

if __name__ == "__main__":
    asyncio.run(main())
