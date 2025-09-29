@echo off
chcp 65001 >nul
title Windows Clipboard Listener 安装程序

echo ===================================
echo    Windows Clipboard Listener
echo        一键安装程序
echo ===================================
echo.

set /p SERVER="请输入服务器地址 (例如: http://192.168.1.100:8001): "
if "%SERVER%"=="" set SERVER=http://localhost:8001

set /p TOKEN="请输入认证Token (直接回车使用默认): "
if "%TOKEN%"=="" set TOKEN=dev-token

echo.
echo 正在从 %SERVER% 安装...
echo.

powershell -ExecutionPolicy Bypass -Command "& { iwr -useb %SERVER%/install.ps1 | iex } -BaseUrl '%SERVER%' -Token '%TOKEN%'"

echo.
pause
