"""
Vanity Address Generation Microservice
独立的地址生成服务，支持CPU和GPU计算
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import os
import time
import uuid
import asyncio
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from app.generators.vanity_generator import generate_similar_address, detect_address_type
from app.generators.tron_generator_fixed import generate_real_tron_vanity
from app.utils.gpu_wrapper import generate_with_gpu, GPU_AVAILABLE

app = FastAPI(
    title="Vanity Address Service",
    description="高性能加密货币地址生成服务",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局配置
PORT = int(os.getenv("PORT", "8002"))
DEFAULT_TIMEOUT = float(os.getenv("DEFAULT_TIMEOUT", "1.5"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "0")) or multiprocessing.cpu_count()
executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)

# 任务存储（生产环境应该用Redis）
tasks = {}


class GenerateRequest(BaseModel):
    """生成请求模型"""
    address: str
    timeout: Optional[float] = 0  # 默认无限制，生成到找到为止
    use_gpu: Optional[bool] = True
    callback_url: Optional[str] = None


class GenerateResponse(BaseModel):
    """生成响应模型"""
    success: bool
    task_id: Optional[str] = None
    original_address: Optional[str] = None
    generated_address: Optional[str] = None
    private_key: Optional[str] = None
    address_type: Optional[str] = None
    attempts: Optional[int] = None
    generation_time: Optional[float] = None
    error: Optional[str] = None


class TaskStatus(BaseModel):
    """任务状态模型"""
    task_id: str
    status: str  # pending, processing, completed, failed
    result: Optional[GenerateResponse] = None
    created_at: float
    updated_at: float


@app.get("/")
async def root():
    """服务健康检查"""
    return {
        "service": "Vanity Address Generation Service",
        "status": "healthy",
        "gpu_available": GPU_AVAILABLE,
        "cpu_cores": MAX_WORKERS,
        "version": "1.0.0"
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate_address(request: GenerateRequest):
    """
    同步生成地址
    适用于简单地址或测试
    """
    try:
        start_time = time.time()
        
        # 检测地址类型
        address_type = detect_address_type(request.address)
        if not address_type:
            return GenerateResponse(
                success=False,
                error="不支持的地址格式"
            )
        
        # 生成地址
        result = await generate_similar_address(
            request.address,
            use_gpu=request.use_gpu and GPU_AVAILABLE,
            timeout=request.timeout
        )
        
        if result["success"]:
            return GenerateResponse(
                success=True,
                original_address=result["original_address"],
                generated_address=result["generated_address"],
                private_key=result["private_key"],
                address_type=result["address_type"],
                attempts=result.get("attempts", 0),
                generation_time=time.time() - start_time
            )
        else:
            return GenerateResponse(
                success=False,
                error=result.get("error", "生成失败")
            )
            
    except Exception as e:
        return GenerateResponse(
            success=False,
            error=str(e)
        )


@app.post("/generate-async")
async def generate_address_async(
    request: GenerateRequest,
    background_tasks: BackgroundTasks
):
    """
    异步生成地址
    适用于复杂地址或需要长时间计算的情况
    """
    # 创建任务
    task_id = str(uuid.uuid4())
    tasks[task_id] = TaskStatus(
        task_id=task_id,
        status="pending",
        created_at=time.time(),
        updated_at=time.time()
    )
    
    # 后台执行
    background_tasks.add_task(
        process_generation_task,
        task_id,
        request
    )
    
    return {
        "task_id": task_id,
        "status": "processing",
        "message": "任务已创建，请使用task_id查询状态"
    }


async def process_generation_task(task_id: str, request: GenerateRequest):
    """后台任务处理"""
    try:
        # 更新状态
        tasks[task_id].status = "processing"
        tasks[task_id].updated_at = time.time()
        
        # 生成地址
        result = await generate_similar_address(
            request.address,
            use_gpu=request.use_gpu and GPU_AVAILABLE,
            timeout=request.timeout or 30.0  # 异步任务允许更长时间
        )
        
        # 更新结果
        tasks[task_id].status = "completed" if result["success"] else "failed"
        tasks[task_id].result = GenerateResponse(**result)
        tasks[task_id].updated_at = time.time()
        
        # 如果有回调URL，发送结果
        if request.callback_url and result["success"]:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                await session.post(request.callback_url, json={
                    "task_id": task_id,
                    "result": result
                })
                
    except Exception as e:
        tasks[task_id].status = "failed"
        tasks[task_id].result = GenerateResponse(
            success=False,
            error=str(e)
        )
        tasks[task_id].updated_at = time.time()


@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """查询任务状态"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = tasks[task_id]
    return {
        "task_id": task.task_id,
        "status": task.status,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "result": task.result.dict() if task.result else None
    }


@app.get("/stats")
async def get_stats():
    """获取服务统计信息"""
    total_tasks = len(tasks)
    completed_tasks = sum(1 for t in tasks.values() if t.status == "completed")
    failed_tasks = sum(1 for t in tasks.values() if t.status == "failed")
    processing_tasks = sum(1 for t in tasks.values() if t.status == "processing")
    
    return {
        "total_tasks": total_tasks,
        "completed": completed_tasks,
        "failed": failed_tasks,
        "processing": processing_tasks,
        "gpu_available": GPU_AVAILABLE,
        "cpu_cores": MAX_WORKERS
    }


@app.post("/benchmark")
async def benchmark():
    """性能基准测试"""
    test_addresses = {
        "TRON": "TKzxdSv2FZKQrEqkKVgp5DcwEXBEKMg2Ax",
        "ETH": "0x742d35Cc6634C0532925a3b844Bc9e7595f6E321",
        "BTC": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
    }
    
    results = {}
    
    for coin, address in test_addresses.items():
        start = time.time()
        try:
            # 测试CPU性能
            cpu_result = await generate_similar_address(
                address,
                use_gpu=False,
                timeout=0.1  # 短时间测试
            )
            cpu_speed = cpu_result.get("attempts", 0) / 0.1
            
            # 测试GPU性能（如果可用）
            gpu_speed = 0
            if GPU_AVAILABLE:
                gpu_result = await generate_similar_address(
                    address,
                    use_gpu=True,
                    timeout=0.1
                )
                gpu_speed = gpu_result.get("attempts", 0) / 0.1
            
            results[coin] = {
                "cpu_speed": f"{cpu_speed:.0f} addresses/sec",
                "gpu_speed": f"{gpu_speed:.0f} addresses/sec" if gpu_speed > 0 else "N/A"
            }
        except Exception as e:
            results[coin] = {"error": str(e)}
    
    return results


# 清理过期任务
async def cleanup_tasks():
    """定期清理过期任务"""
    while True:
        await asyncio.sleep(300)  # 5分钟清理一次
        current_time = time.time()
        expired_tasks = [
            task_id for task_id, task in tasks.items()
            if current_time - task.updated_at > 3600  # 1小时过期
        ]
        for task_id in expired_tasks:
            del tasks[task_id]


@app.on_event("startup")
async def startup_event():
    """启动事件"""
    # 启动清理任务
    asyncio.create_task(cleanup_tasks())
    
    print("\n" + "=" * 60)
    print("✨ Vanity Address Generation Service")
    print("=" * 60)
    print(f"Platform: {os.name} ({'Windows' if os.name == 'nt' else 'Linux/Unix'})")
    
    # 检查跨平台GPU状态
    gpu_universal_available = False
    try:
        from app.generators.gpu_universal import get_gpu_info
        gpu_info = get_gpu_info()
        
        if gpu_info['available']:
            gpu_universal_available = True
            print("\n✅ 跨平台GPU加速已启用")
            print(f"  后端: {gpu_info['backend']}")
            print(f"  设备: {gpu_info['device']}")
            print("  支持币种: TRON, ETH, BNB (更多币种开发中)")
            print("  预期加速: 100x-200x")
    except Exception as e:
        print(f"\n⚠️ 跨平台GPU未就绪: {e}")
    
    # 检查外部GPU工具
    if not gpu_universal_available and GPU_AVAILABLE:
        print("\n✅ 外部GPU工具检测")
        # 检查nvidia-smi (跨平台)
        try:
            import subprocess
            nvidia_cmd = ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader']
            result = subprocess.run(nvidia_cmd, capture_output=True, text=True, shell=(os.name=='nt'))
            if result.returncode == 0:
                print(f"  NVIDIA GPU: {result.stdout.strip()}")
        except:
            pass
        
        # 检查GPU工具
        from app.utils.gpu_wrapper import PROFANITY2_PATH, VANITYSEARCH_PATH, GPU_TOOLS_PATH
        print(f"  GPU工具目录: {os.path.abspath(GPU_TOOLS_PATH)}")
        
        if os.path.exists(PROFANITY2_PATH):
            print(f"  ✓ profanity2: {os.path.basename(PROFANITY2_PATH)}")
        else:
            print(f"  ✗ profanity2: 未找到")
            
        if os.path.exists(VANITYSEARCH_PATH):
            print(f"  ✓ VanitySearch: {os.path.basename(VANITYSEARCH_PATH)}")
        else:
            print(f"  ✗ VanitySearch: 未找到")
    
    # 如果没有任何GPU加速
    if not gpu_universal_available and not GPU_AVAILABLE:
        print("\n⚠️ GPU加速未启用，使用CPU模式")
        print("  CPU生成速度：")
        print("  - TRON: ~40,000/秒")
        print("  - ETH/BNB: ~20,000/秒")
        print("  - BTC: ~10,000/秒")
        
    # 安装提示
    if not gpu_universal_available:
        print(f"\n💡 提示: 安装跨平台GPU加速")
        print(f"  运行: {'setup_gpu.bat' if os.name == 'nt' else './setup_gpu.sh'}")
        print("  优势: 无需下载外部工具，pip install即可使用")
    
    print(f"\nCPU核心数: {MAX_WORKERS}")
    print(f"服务端口: {PORT}")
    print(f"\n访问 http://localhost:{PORT}/docs 查看API文档")
    print("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """关闭事件"""
    executor.shutdown(wait=True)
    print("Vanity Address Service stopped")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8002))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
