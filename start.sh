#!/bin/bash

echo "=========================================="
echo "启动 Windows Clipboard Listener"
echo "=========================================="
echo

# 启动地址生成服务
echo "[1/3] 启动地址生成服务..."
cd vanity-service
python3 main.py &
VANITY_PID=$!
cd ..

# 启动主API服务
echo "[2/3] 启动主API服务..."
cd server
if [ -d "../venv" ]; then
    source ../venv/bin/activate
fi
python3 main.py &
API_PID=$!
cd ..

# 启动管理界面
echo "[3/3] 启动管理界面..."
cd admin
npm run dev &
ADMIN_PID=$!
cd ..

echo
echo "=========================================="
echo "所有服务已启动！"
echo
echo "- 地址生成服务: http://localhost:8002 (PID: $VANITY_PID)"
echo "- 主API服务: http://localhost:8001 (PID: $API_PID)"
echo "- 管理界面: http://localhost:5173 (PID: $ADMIN_PID)"
echo
echo "按 Ctrl+C 停止所有服务"
echo "=========================================="

# 等待中断信号
trap "kill $VANITY_PID $API_PID $ADMIN_PID; exit" INT
wait
