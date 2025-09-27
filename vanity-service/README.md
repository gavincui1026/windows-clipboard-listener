# Vanity Address Generation Service

高性能加密货币地址生成微服务，支持TRON、BTC、ETH、BNB等主流币种。

## 特性

- ✅ 支持多种加密货币地址生成
- ✅ CPU多进程并行计算
- ✅ GPU加速支持（可选）
- ✅ RESTful API接口
- ✅ 异步任务处理
- ✅ Docker容器化部署

## 支持的币种

| 币种 | CPU支持 | GPU支持 | 速度（CPU） | 速度（GPU） |
|------|---------|---------|-------------|-------------|
| TRON | ✅ | ✅ | 4万/秒 | 3000万/秒 |
| ETH  | ✅ | ✅ | 2万/秒 | 5000万/秒 |
| BNB  | ✅ | ✅ | 2万/秒 | 5000万/秒 |
| BTC  | ⚠️  | ✅ | 1万/秒 | 2000万/秒 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务

```bash
# 开发模式
uvicorn main:app --reload --port 8001

# 生产模式
uvicorn main:app --host 0.0.0.0 --port 8001 --workers 4
```

### 3. Docker部署

```bash
# 构建镜像
docker build -t vanity-service .

# 运行容器
docker run -d -p 8001:8001 vanity-service

# 或使用docker-compose
docker-compose up -d
```

## API使用

### 健康检查

```bash
GET /
```

### 同步生成（快速模式）

```bash
POST /generate
Content-Type: application/json

{
  "address": "TKzxdSv2FZKQrEqkKVgp5DcwEXBEKMg2Ax",
  "timeout": 1.5,
  "use_gpu": false
}
```

响应：
```json
{
  "success": true,
  "original_address": "TKzxdSv2FZKQrEqkKVgp5DcwEXBEKMg2Ax",
  "generated_address": "TKz8pYm3QrkqNBB5DcwEXBEKMg6yE82Ax",
  "private_key": "5JYkZjmN7PX2MjRc...",
  "address_type": "TRON",
  "attempts": 523421,
  "generation_time": 1.23
}
```

### 异步生成（复杂地址）

```bash
POST /generate-async
Content-Type: application/json

{
  "address": "0x742d35Cc6634C0532925a3b844Bc9e7595f6E321",
  "timeout": 30,
  "use_gpu": true,
  "callback_url": "https://your-api.com/callback"
}
```

响应：
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "message": "任务已创建，请使用task_id查询状态"
}
```

### 查询任务状态

```bash
GET /task/{task_id}
```

### 性能测试

```bash
POST /benchmark
```

## GPU加速设置

### 1. 安装GPU工具

```bash
# 创建工具目录
mkdir gpu_tools

# 下载profanity2 (ETH/BNB)
wget https://github.com/1inch/profanity2/releases/latest/download/profanity2
chmod +x profanity2
mv profanity2 gpu_tools/

# 下载VanitySearch (BTC)
wget https://github.com/JeanLucPons/VanitySearch/releases/latest/download/VanitySearch
chmod +x VanitySearch
mv VanitySearch gpu_tools/
```

### 2. 设置环境变量

```bash
export GPU_TOOLS_PATH=/path/to/gpu_tools
```

## 性能优化

### CPU优化

1. 使用`coincurve`替代`ecdsa`（快5-10倍）
```bash
pip install coincurve
```

2. 增加进程数
```python
MAX_WORKERS = multiprocessing.cpu_count() * 2
```

### GPU优化

1. 使用NVIDIA GPU（推荐RTX 3060及以上）
2. 安装CUDA驱动
3. 使用专业GPU工具

## 集成到主服务

### 修改主服务调用

```python
# 在main.py中
import aiohttp

VANITY_SERVICE_URL = "http://vanity-service:8001"

async def generate_similar_address(address: str):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{VANITY_SERVICE_URL}/generate",
            json={"address": address}
        ) as resp:
            return await resp.json()
```

## 部署架构

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   前端      │────▶│   主服务     │────▶│ Vanity服务  │
│  (Vercel)   │     │  (Railway)   │     │  (CPU/GPU)  │
└─────────────┘     └──────────────┘     └─────────────┘
```

## 成本估算

| 部署方式 | 月成本 | 性能 | 适用场景 |
|----------|--------|------|----------|
| Railway CPU | $5 | 中等 | 低频使用 |
| VPS + Docker | $10 | 高 | 中频使用 |
| GPU云服务 | $50+ | 极高 | 高频使用 |

## 故障排除

### 地址生成失败
- 检查地址格式是否正确
- 增加超时时间
- 检查CPU/GPU资源

### GPU不可用
- 检查CUDA驱动
- 确认GPU工具路径
- 查看错误日志

### 性能问题
- 使用更多CPU核心
- 启用GPU加速
- 优化地址模式

## 许可证

MIT License
