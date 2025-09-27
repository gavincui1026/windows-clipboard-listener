# 独立部署指南

## 架构说明

本项目支持完全独立的部署架构：

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  前端 (Vue)     │────▶│  主API服务        │────▶│ 地址生成服务     │
│  端口: 5173     │HTTP │  端口: 8001      │HTTP │  端口: 8002     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                           │
                               ▼                           ▼
                        ┌──────────────┐           ┌──────────────┐
                        │   MySQL DB    │           │  GPU (可选)   │
                        └──────────────┘           └──────────────┘
```

## 部署方式

### 方式1：完整Docker部署（推荐）

```bash
# 一键启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 方式2：独立部署各服务

#### 1. 部署地址生成服务（可部署到GPU服务器）

```bash
# 在GPU服务器上
cd vanity-service
docker build -t vanity-service .
docker run -d \
  -p 8002:8002 \
  -v $(pwd)/gpu_tools:/app/gpu_tools \
  --gpus all \  # 如果有GPU
  vanity-service
```

#### 2. 部署主API服务

```bash
# 在主服务器上
cd server
docker build -t clipboard-api .
docker run -d \
  -p 8001:8001 \
  -e VANITY_SERVICE_URL=http://gpu-server:8002 \  # 指向GPU服务器
  -e DB_HOST=your-db-host \
  -e DB_USER=your-db-user \
  -e DB_PASSWORD=your-db-pass \
  clipboard-api
```

#### 3. 部署前端（可使用CDN）

```bash
cd admin
npm run build
# 将dist目录部署到任何静态文件服务器
```

## 独立部署示例

### 场景1：主服务在阿里云，GPU服务在AWS

**GPU服务器（AWS）：**
```bash
# 启动地址生成服务
cd vanity-service
export GPU_TOOLS_PATH=/opt/gpu_tools
python main.py --host 0.0.0.0 --port 8002
```

**主服务器（阿里云）：**
```bash
# 配置环境变量
export VANITY_SERVICE_URL=http://aws-gpu-server.com:8002

# 启动主服务
cd server
python main.py
```

### 场景2：使用Kubernetes部署

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
      nodeSelector:
        gpu: "true"  # 部署到GPU节点
      containers:
      - name: vanity-service
        image: vanity-service:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        ports:
        - containerPort: 8002
```

### 场景3：混合云部署

1. **主服务**：部署在私有云
2. **地址生成服务**：部署在公有云GPU实例
3. **数据库**：使用云数据库服务

## 配置说明

### 主服务配置

```env
# server/.env
VANITY_SERVICE_URL=http://gpu-service.example.com:8002
DB_HOST=rds.aliyuncs.com
DB_USER=clipboard_user
DB_PASSWORD=secure_password
```

### 地址生成服务配置

```env
# vanity-service/.env
PORT=8002
GPU_TOOLS_PATH=/app/gpu_tools
ENABLE_GPU=true
MAX_WORKERS=8
```

## 负载均衡配置

使用Nginx做地址生成服务的负载均衡：

```nginx
upstream vanity_backends {
    least_conn;
    server gpu1.example.com:8002 weight=3;  # GPU服务器
    server gpu2.example.com:8002 weight=3;  # GPU服务器
    server cpu1.example.com:8002 weight=1;  # CPU备用
}

server {
    listen 80;
    server_name vanity-lb.example.com;

    location / {
        proxy_pass http://vanity_backends;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
}
```

## 监控和健康检查

### 健康检查端点

- 主服务：`GET http://localhost:8001/`
- 地址生成服务：`GET http://localhost:8002/`

### Prometheus监控

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'clipboard-api'
    static_configs:
      - targets: ['api-server:8001']
  
  - job_name: 'vanity-service'
    static_configs:
      - targets: ['gpu-server:8002']
```

## 性能优化建议

1. **地址生成服务**可部署多个实例，通过负载均衡分发请求
2. **GPU服务器**选择高性能GPU（如RTX 4090或A100）
3. **缓存策略**：对常见地址模式进行缓存
4. **异步处理**：复杂地址使用异步任务处理

## 成本优化

1. **按需GPU**：使用云服务的竞价实例
2. **混合计算**：简单地址用CPU，复杂地址用GPU
3. **预生成池**：预先生成常见模式的地址

## 故障转移

主服务会自动处理地址生成服务不可用的情况：

```python
# 服务降级逻辑已内置
if not await client.health_check():
    return {"success": False, "error": "地址生成服务不可用"}
```

## 安全建议

1. 使用HTTPS通信
2. API密钥认证
3. 网络隔离（VPC）
4. 限流保护
