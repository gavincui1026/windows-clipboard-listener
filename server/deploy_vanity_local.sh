#!/bin/bash
# 一键部署本地vanity服务并修复连接问题

set -e

echo "=== 部署本地Vanity服务 ==="
echo

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否在项目根目录
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}错误: 请在项目根目录运行此脚本${NC}"
    echo "当前目录: $(pwd)"
    exit 1
fi

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo -e "${RED}错误: Docker未安装${NC}"
    echo "请先安装Docker: https://docs.docker.com/engine/install/"
    exit 1
fi

# 检查Docker Compose
if ! command -v docker-compose &> /dev/null; then
    # 尝试使用docker compose
    if docker compose version &> /dev/null; then
        alias docker-compose='docker compose'
    else
        echo -e "${RED}错误: Docker Compose未安装${NC}"
        echo "请先安装Docker Compose"
        exit 1
    fi
fi

echo "1. 检查vanity服务状态..."
if docker ps | grep -q vanity-service; then
    echo -e "${GREEN}✓ Vanity服务已在运行${NC}"
else
    echo -e "${YELLOW}启动vanity服务...${NC}"
    docker-compose up -d vanity-service
    
    # 等待服务启动
    echo "等待服务启动..."
    sleep 5
    
    if docker ps | grep -q vanity-service; then
        echo -e "${GREEN}✓ Vanity服务启动成功${NC}"
    else
        echo -e "${RED}✗ Vanity服务启动失败${NC}"
        echo "查看日志: docker-compose logs vanity-service"
        exit 1
    fi
fi

echo
echo "2. 配置环境变量..."

# 创建.env文件
ENV_FILE="server/.env"
if [ -f "$ENV_FILE" ]; then
    # 备份现有文件
    cp "$ENV_FILE" "$ENV_FILE.bak"
    echo -e "${YELLOW}已备份现有配置到: $ENV_FILE.bak${NC}"
fi

# 检查是否已有正确的配置
if [ -f "$ENV_FILE" ] && grep -q "VANITY_SERVICE_URL=http://localhost:8002" "$ENV_FILE"; then
    echo -e "${GREEN}✓ 环境变量已正确配置${NC}"
else
    # 更新或添加VANITY_SERVICE_URL
    if [ -f "$ENV_FILE" ]; then
        # 删除旧的VANITY_SERVICE_URL配置
        sed -i '/^VANITY_SERVICE_URL=/d' "$ENV_FILE"
    else
        # 从模板创建
        if [ -f "server/env.example" ]; then
            cp "server/env.example" "$ENV_FILE"
        else
            touch "$ENV_FILE"
        fi
    fi
    
    # 添加本地vanity服务配置
    echo "" >> "$ENV_FILE"
    echo "# Vanity服务配置（使用本地服务）" >> "$ENV_FILE"
    echo "VANITY_SERVICE_URL=http://localhost:8002" >> "$ENV_FILE"
    
    echo -e "${GREEN}✓ 已更新环境变量配置${NC}"
fi

echo
echo "3. 测试连接..."
cd server

# 运行连接测试
if python3 test_vanity_connection.py 2>/dev/null | grep -q "localhost:8002.*✓"; then
    echo -e "${GREEN}✓ 本地vanity服务连接成功${NC}"
else
    echo -e "${YELLOW}⚠ 连接测试失败，尝试其他方法...${NC}"
fi

cd ..

echo
echo "4. 重启API服务..."
if docker ps | grep -q "clipboard.*api"; then
    docker-compose restart api
    echo -e "${GREEN}✓ API服务已重启${NC}"
else
    echo -e "${YELLOW}启动API服务...${NC}"
    docker-compose up -d api
fi

# 等待服务启动
sleep 5

echo
echo "5. 验证部署..."

# 检查服务健康状态
API_HEALTH=$(curl -s http://localhost:8001/health 2>/dev/null || echo "failed")
VANITY_HEALTH=$(curl -s http://localhost:8002/ 2>/dev/null || echo "failed")

if [[ "$API_HEALTH" != "failed" ]]; then
    echo -e "${GREEN}✓ API服务正常${NC}"
else
    echo -e "${RED}✗ API服务异常${NC}"
fi

if [[ "$VANITY_HEALTH" != "failed" ]]; then
    echo -e "${GREEN}✓ Vanity服务正常${NC}"
else
    echo -e "${RED}✗ Vanity服务异常${NC}"
fi

echo
echo "=== 部署完成 ==="
echo
echo "后续操作建议："
echo "1. 查看服务日志: docker-compose logs -f"
echo "2. 测试vanity生成: curl -X POST http://localhost:8002/generate -H 'Content-Type: application/json' -d '{\"address\":\"TKzxdSv2FZKQrEqkKVgp5DcwEXBEKMg2Ax\"}'"
echo "3. 如果仍有问题，运行: cd server && python3 fix_vanity_connection.py"
echo
echo -e "${GREEN}提示: 已将vanity服务配置为使用本地部署，避免了网络连接问题${NC}"
