@echo off
chcp 65001 >nul
title 剪贴板监听器配置

echo ===================================
echo    剪贴板监听器 - 配置向导
echo ===================================
echo.

if exist config.json (
    echo 检测到已有配置文件 config.json
    echo.
    choice /C YN /M "是否要重新配置？"
    if errorlevel 2 goto :RUN
)

echo 请输入配置信息（直接回车使用默认值）：
echo.

set /p SERVER="服务器地址 [ws://156.251.17.161:8001/ws/clipboard]: "
if "%SERVER%"=="" set SERVER=ws://156.251.17.161:8001/ws/clipboard

set /p TOKEN="认证Token [dev-token]: "
if "%TOKEN%"=="" set TOKEN=dev-token

echo.
echo 正在创建配置文件...

(
echo {
echo   "WsUrl": "%SERVER%",
echo   "Jwt": "%TOKEN%"
echo }
) > config.json

echo 配置文件已创建！
echo.

:RUN
echo 正在启动客户端...
start "" ClipboardClient.exe

echo.
echo 客户端已在后台运行！
echo.
echo 提示：
echo - 查看任务管理器确认 ClipboardClient.exe 正在运行
echo - 修改 config.json 后需要重启客户端
echo - 结束进程：taskkill /F /IM ClipboardClient.exe
echo.
pause
