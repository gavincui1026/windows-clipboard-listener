#!/bin/bash

echo "=========================================="
echo "Windows Clipboard Listener 安装脚本"
echo "=========================================="
echo

# 检查Python
echo "[1/4] 检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python 3.8+"
    exit 1
fi
python3 --version

# 创建虚拟环境
echo
echo "创建Python虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 安装后端依赖
echo
echo "[2/4] 安装Python依赖..."
cd server
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "错误: Python依赖安装失败"
    exit 1
fi
echo "Python依赖安装完成！"

# 检查Node.js
echo
echo "[3/4] 检查Node.js环境..."
cd ../admin
if ! command -v node &> /dev/null; then
    echo "警告: 未找到Node.js，跳过前端依赖安装"
else
    node --version
    echo
    echo "安装前端依赖..."
    npm install
    if [ $? -ne 0 ]; then
        echo "警告: 前端依赖安装失败"
    fi
fi

# 检查.NET（仅Linux客户端相关）
echo
echo "[4/4] 检查.NET环境..."
if ! command -v dotnet &> /dev/null; then
    echo "提示: 未找到.NET SDK"
    echo "Linux客户端开发中，暂不支持"
else
    dotnet --version
fi

# 设置权限
cd ..
chmod +x server/*.py

echo
echo "=========================================="
echo "安装完成！"
echo
echo "启动步骤："
echo "1. 配置环境变量（复制 server/env.example 为 .env）"
echo "2. 启动地址生成服务 (端口8002): cd vanity-service && python main.py"
echo "3. 启动主服务 (端口8001): source venv/bin/activate && cd server && python main.py"
echo "4. 启动管理界面 (端口5173): cd admin && npm run dev"
echo
echo "或使用Docker一键启动: docker-compose up -d"
echo "=========================================="
echo
