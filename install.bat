@echo off
echo ==========================================
echo Windows Clipboard Listener 安装脚本
echo ==========================================
echo.

REM 检查Python
echo [1/4] 检查Python环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)
python --version

REM 安装后端依赖
echo.
echo [2/4] 安装Python依赖...
cd server
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo 错误: Python依赖安装失败
    pause
    exit /b 1
)
echo Python依赖安装完成！

REM 检查Node.js
echo.
echo [3/4] 检查Node.js环境...
cd ..\admin
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 警告: 未找到Node.js，跳过前端依赖安装
    goto :dotnet
)
node --version

REM 安装前端依赖
echo.
echo 安装前端依赖...
npm install
if %errorlevel% neq 0 (
    echo 警告: 前端依赖安装失败
)

:dotnet
REM 检查.NET
echo.
echo [4/4] 检查.NET环境...
cd ..\client\ClipboardClient
dotnet --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 警告: 未找到.NET SDK，跳过客户端编译
    goto :finish
)
dotnet --version

REM 编译客户端
echo.
echo 编译客户端...
dotnet build
if %errorlevel% neq 0 (
    echo 警告: 客户端编译失败
)

:finish
cd ..\..
echo.
echo ==========================================
echo 安装完成！
echo.
echo 启动步骤：
echo 1. 配置环境变量（复制 server\env.example 为 .env）
echo 2. 启动地址生成服务 (端口8002): cd vanity-service && python main.py
echo 3. 启动主服务 (端口8001): cd server && python main.py
echo 4. 启动管理界面 (端口5173): cd admin && npm run dev
echo.
echo 或使用Docker一键启动: docker-compose up -d
echo ==========================================
echo.
pause
