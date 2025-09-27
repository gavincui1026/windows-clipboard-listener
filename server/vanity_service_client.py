"""
Vanity Service客户端
用于主服务调用独立的地址生成微服务
"""
import os
import aiohttp
import asyncio
from typing import Dict, Optional


class VanityServiceClient:
    """地址生成服务客户端"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("VANITY_SERVICE_URL", "http://localhost:8002")
        self.session = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            async with self.session.get(f"{self.base_url}/") as resp:
                return resp.status == 200
        except:
            return False
    
    async def generate_sync(
        self,
        address: str,
        timeout: float = 0,
        use_gpu: bool = True
    ) -> Dict:
        """同步生成地址（等待结果）"""
        try:
            # 如果timeout=0，不设置HTTP超时
            http_timeout = None if timeout == 0 else aiohttp.ClientTimeout(total=timeout + 60)
            
            async with self.session.post(
                f"{self.base_url}/generate",
                json={
                    "address": address,
                    "timeout": timeout,
                    "use_gpu": use_gpu
                },
                timeout=http_timeout
            ) as resp:
                # 检查响应状态
                if resp.status != 200:
                    return {
                        "success": False,
                        "error": f"服务返回错误: {resp.status}"
                    }
                
                # 解析JSON响应
                try:
                    data = await resp.json()
                    # 确保返回的数据包含必要字段
                    if isinstance(data, dict) and 'success' in data:
                        return data
                    else:
                        return {
                            "success": False,
                            "error": f"响应格式错误: {data}"
                        }
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"解析响应失败: {str(e)}"
                    }
                    
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "生成超时"
            }
        except aiohttp.ClientError as e:
            return {
                "success": False,
                "error": f"连接错误: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"未知错误: {str(e)}"
            }
    
    async def generate_async(
        self,
        address: str,
        timeout: float = 30.0,
        use_gpu: bool = True,
        callback_url: str = None
    ) -> Dict:
        """异步生成地址（返回任务ID）"""
        try:
            async with self.session.post(
                f"{self.base_url}/generate-async",
                json={
                    "address": address,
                    "timeout": timeout,
                    "use_gpu": use_gpu,
                    "callback_url": callback_url
                }
            ) as resp:
                return await resp.json()
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_task_status(self, task_id: str) -> Dict:
        """查询任务状态"""
        try:
            async with self.session.get(
                f"{self.base_url}/task/{task_id}"
            ) as resp:
                if resp.status == 404:
                    return {"error": "任务不存在"}
                return await resp.json()
        except Exception as e:
            return {"error": str(e)}
    
    async def benchmark(self) -> Dict:
        """性能测试"""
        try:
            async with self.session.post(
                f"{self.base_url}/benchmark"
            ) as resp:
                return await resp.json()
        except Exception as e:
            return {"error": str(e)}
    
    async def get_stats(self) -> Dict:
        """获取服务统计"""
        try:
            async with self.session.get(
                f"{self.base_url}/stats"
            ) as resp:
                return await resp.json()
        except Exception as e:
            return {"error": str(e)}


# 使用示例
async def example():
    """使用示例"""
    async with VanityServiceClient() as client:
        # 健康检查
        if not await client.health_check():
            print("Vanity服务不可用")
            return
        
        # 同步生成（快速）
        result = await client.generate_sync(
            "TKzxdSv2FZKQrEqkKVgp5DcwEXBEKMg2Ax",
            timeout=1.5
        )
        
        if result["success"]:
            print(f"生成成功: {result['generated_address']}")
            print(f"私钥: {result['private_key']}")
        else:
            print(f"生成失败: {result['error']}")
        
        # 异步生成（复杂地址）
        task = await client.generate_async(
            "0x742d35Cc6634C0532925a3b844Bc9e7595f6E321",
            timeout=30.0,
            use_gpu=True
        )
        
        if "task_id" in task:
            print(f"任务创建: {task['task_id']}")
            
            # 轮询任务状态
            while True:
                await asyncio.sleep(1)
                status = await client.get_task_status(task['task_id'])
                
                if status.get("status") == "completed":
                    print(f"任务完成: {status['result']}")
                    break
                elif status.get("status") == "failed":
                    print(f"任务失败: {status}")
                    break


if __name__ == "__main__":
    asyncio.run(example())
