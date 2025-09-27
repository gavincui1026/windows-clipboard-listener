#!/bin/bash

# Vanity Service 部署脚本

echo "==================================="
echo "Vanity Address Service 部署脚本"
echo "==================================="

# 检查Docker
if ! command -v docker &> /dev/null; then
    echo "错误: Docker未安装"
    exit 1
fi

# 选择部署模式
echo "请选择部署模式:"
echo "1. CPU模式（标准）"
echo "2. GPU模式（需要NVIDIA GPU）"
read -p "选择 (1/2): " mode

# 构建镜像
echo "构建Docker镜像..."
if [ "$mode" == "2" ]; then
    docker build --target gpu -t vanity-service:gpu .
else
    docker build -t vanity-service:latest .
fi

# 创建GPU工具目录
mkdir -p gpu_tools

# 下载GPU工具（可选）
read -p "是否下载GPU工具? (y/n): " download_tools
if [ "$download_tools" == "y" ]; then
    echo "下载GPU工具..."
    
    # profanity2
    if [ ! -f "gpu_tools/profanity2" ]; then
        echo "下载profanity2..."
        wget -O gpu_tools/profanity2 https://github.com/1inch/profanity2/releases/latest/download/profanity2
        chmod +x gpu_tools/profanity2
    fi
    
    # VanitySearch
    if [ ! -f "gpu_tools/VanitySearch" ]; then
        echo "下载VanitySearch..."
        wget -O gpu_tools/VanitySearch https://github.com/JeanLucPons/VanitySearch/releases/latest/download/VanitySearch
        chmod +x gpu_tools/VanitySearch
    fi
fi

# 停止旧容器
echo "停止旧容器..."
docker stop vanity-service 2>/dev/null || true
docker rm vanity-service 2>/dev/null || true

# 启动新容器
echo "启动服务..."
if [ "$mode" == "2" ]; then
    # GPU模式
    docker run -d \
        --name vanity-service \
        --gpus all \
        -p 8001:8001 \
        -v $(pwd)/gpu_tools:/app/gpu_tools \
        -e GPU_TOOLS_PATH=/app/gpu_tools \
        --restart unless-stopped \
        vanity-service:gpu
else
    # CPU模式
    docker run -d \
        --name vanity-service \
        -p 8001:8001 \
        -v $(pwd)/gpu_tools:/app/gpu_tools \
        -e GPU_TOOLS_PATH=/app/gpu_tools \
        --restart unless-stopped \
        vanity-service:latest
fi

# 检查状态
sleep 3
if docker ps | grep vanity-service > /dev/null; then
    echo "✅ 服务启动成功！"
    echo "访问: http://localhost:8001"
    echo ""
    docker logs vanity-service --tail 20
else
    echo "❌ 服务启动失败"
    docker logs vanity-service
    exit 1
fi
