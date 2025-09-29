#!/usr/bin/env python3
"""
快速测试vanity服务连接
"""
import os
import sys
import asyncio
import aiohttp
from typing import List, Tuple

# 测试URL列表
TEST_URLS = [
    "http://localhost:8002",
    "http://127.0.0.1:8002",
    "http://vanity-service:8002",  # Docker内部网络
    "https://trainers-pads-switches-links.trycloudflare.com",
]

async def test_url(session: aiohttp.ClientSession, url: str) -> Tuple[str, bool, str]:
    """测试单个URL"""
    try:
        async with session.get(f"{url}/", timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                return url, True, "✓ 连接成功"
            else:
                return url, False, f"✗ HTTP {resp.status}"
    except aiohttp.ClientConnectorCertificateError as e:
        return url, False, "✗ SSL证书验证失败"
    except aiohttp.ClientError as e:
            error_msg = str(e)
            if "Cannot connect" in error_msg:
                return url, False, "✗ 无法连接"
            elif "Network is unreachable" in error_msg:
                return url, False, "✗ 网络不可达"
            elif "Name or service not known" in error_msg:
                return url, False, "✗ DNS解析失败"
            else:
                return url, False, f"✗ {error_msg[:50]}..."
    except Exception as e:
        return url, False, f"✗ 错误: {str(e)[:50]}..."

async def test_all_urls(custom_url: str = None) -> List[Tuple[str, bool, str]]:
    """测试所有URL"""
    urls = TEST_URLS.copy()
    
    # 添加环境变量中的URL
    env_url = os.getenv("VANITY_SERVICE_URL")
    if env_url and env_url not in urls:
        urls.append(env_url)
    
    # 添加自定义URL
    if custom_url and custom_url not in urls:
        urls.append(custom_url)
    
    results = []
    # 创建不验证SSL的会话（用于测试）
    connector = aiohttp.TCPConnector(verify_ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [test_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
    
    return results

def print_results(results: List[Tuple[str, bool, str]]):
    """打印测试结果"""
    print("\n" + "="*60)
    print("Vanity服务连接测试结果")
    print("="*60)
    
    success_count = sum(1 for _, success, _ in results if success)
    
    for url, success, msg in results:
        status_icon = "✅" if success else "❌"
        print(f"{status_icon} {url}")
        print(f"   {msg}")
        print()
    
    print("-"*60)
    print(f"测试完成: {success_count}/{len(results)} 个URL可以连接")
    
    # 提供建议
    print("\n建议:")
    if any(success and "localhost" in url for url, success, _ in results):
        print("✓ 本地vanity服务可用，建议使用:")
        print("  export VANITY_SERVICE_URL=http://localhost:8002")
        print("  或在.env文件中设置: VANITY_SERVICE_URL=http://localhost:8002")
    elif any(success for _, success, _ in results):
        success_url = next(url for url, success, _ in results if success)
        print(f"✓ 发现可用的服务: {success_url}")
        print(f"  export VANITY_SERVICE_URL={success_url}")
    else:
        print("✗ 没有可用的vanity服务")
        print("  请确保vanity服务正在运行:")
        print("  docker-compose up -d vanity-service")
        print("  或")
        print("  cd vanity-service && python main.py")

async def main():
    """主函数"""
    # 检查是否有自定义URL参数
    custom_url = None
    if len(sys.argv) > 1:
        custom_url = sys.argv[1]
        print(f"添加自定义URL测试: {custom_url}")
    
    # 运行测试
    results = await test_all_urls(custom_url)
    
    # 打印结果
    print_results(results)

if __name__ == "__main__":
    asyncio.run(main())
