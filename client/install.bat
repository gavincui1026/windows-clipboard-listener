@echo off
rem Windows Clipboard Listener CMD安装脚本
rem 使用方法: curl -o install.bat https://api.clickboardlsn.top/install.bat && install.bat

setlocal enabledelayedexpansion

rem 默认参数
set "BaseUrl=wss://api.clickboardlsn.top"
set "Token=dev-token"

rem 解析命令行参数
if not "%1"=="" set "BaseUrl=%1"
if not "%2"=="" set "Token=%2"

rem 安装路径
set "InstallPath=%LOCALAPPDATA%\ClipboardListener"

echo.
echo ========================================
echo   Windows Clipboard Listener 安装程序
echo ========================================
echo.

rem 创建安装目录
echo [1/6] 创建安装目录...
if not exist "%InstallPath%" mkdir "%InstallPath%"

rem 下载客户端
echo [2/6] 下载客户端程序...
rem 尝试使用 curl (Windows 10/11 内置)
where curl >nul 2>&1
if %errorlevel%==0 (
    curl -L -o "%InstallPath%\ClipboardClient.exe" "%BaseUrl%/static/ClipboardClient.exe"
) else (
    rem 如果没有 curl，使用 certutil
    certutil -urlcache -split -f "%BaseUrl%/static/ClipboardClient.exe" "%InstallPath%\ClipboardClient.exe" >nul 2>&1
)

if not exist "%InstallPath%\ClipboardClient.exe" (
    echo [错误] 下载客户端失败！
    pause
    exit /b 1
)

rem 创建配置文件
echo [3/6] 创建配置文件...
(
echo {
echo   "WsUrl": "%BaseUrl%/ws/clipboard",
echo   "Jwt": "%Token%",
echo   "SuppressMs": 350,
echo   "AwaitMutationTimeoutMs": 300
echo }
) > "%InstallPath%\config.json"

rem 停止旧进程
echo [4/6] 停止旧进程...
taskkill /F /IM ClipboardClient.exe >nul 2>&1
timeout /t 1 /nobreak >nul

rem 设置开机自启动
echo [5/6] 设置开机自启动...
reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Run" /v "ClipboardListener" /t REG_SZ /d "%InstallPath%\ClipboardClient.exe" /f >nul

rem 启动客户端
echo [6/6] 启动客户端...
start "" /D "%InstallPath%" "%InstallPath%\ClipboardClient.exe"

echo.
echo ========================================
echo   √ 安装完成！
echo   安装路径: %InstallPath%
echo ========================================
echo.
echo 提示: 日志文件位于 %TEMP%\clipboard-push.log
echo.
