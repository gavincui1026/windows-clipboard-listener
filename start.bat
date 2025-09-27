@echo off
echo ==========================================
echo 启动 Windows Clipboard Listener
echo ==========================================
echo.

echo [1/3] 启动地址生成服务...
start cmd /k "cd vanity-service && python main.py"

echo [2/3] 启动主API服务...
start cmd /k "cd server && python main.py"

echo [3/3] 启动管理界面...
start cmd /k "cd admin && npm run dev"

echo.
echo ==========================================
echo 所有服务已启动！
echo.
echo - 地址生成服务: http://localhost:8002
echo - 主API服务: http://localhost:8001
echo - 管理界面: http://localhost:5173
echo.
echo 按 Ctrl+C 停止服务
echo ==========================================
pause
