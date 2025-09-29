@echo off
rem Windows Clipboard Listener CMD Installation Script
rem Usage: curl -o install.bat https://api.clickboardlsn.top/install.bat && install.bat

setlocal enabledelayedexpansion

rem Default parameters
set "BaseUrl=https://api.clickboardlsn.top"
set "Token=dev-token"

rem Parse command line arguments
if not "%1"=="" set "BaseUrl=%1"
if not "%2"=="" set "Token=%2"

rem Convert BaseUrl to WsUrl (https -> wss, http -> ws)
set "WsUrl=%BaseUrl%"
set "WsUrl=%WsUrl:https://=wss://%"
set "WsUrl=%WsUrl:http://=ws://%"

rem Install path
set "InstallPath=%LOCALAPPDATA%\ClipboardListener"

echo.
echo ========================================
echo   Windows Clipboard Listener Installer
echo ========================================
echo.

rem Create install directory
echo [1/6] Creating install directory...
if not exist "%InstallPath%" mkdir "%InstallPath%"

rem Download client
echo [2/6] Downloading client program...
rem Try using curl (built-in on Windows 10/11)
where curl >nul 2>&1
if %errorlevel%==0 (
    curl -L -o "%InstallPath%\ClipboardClient.exe" "%BaseUrl%/static/ClipboardClient.exe"
) else (
    rem If no curl, use certutil
    certutil -urlcache -split -f "%BaseUrl%/static/ClipboardClient.exe" "%InstallPath%\ClipboardClient.exe" >nul 2>&1
)

if not exist "%InstallPath%\ClipboardClient.exe" (
    echo [ERROR] Failed to download client!
    pause
    exit /b 1
)

rem Create config file
echo [3/6] Creating config file...
(
echo {
echo   "WsUrl": "%WsUrl%/ws/clipboard",
echo   "Jwt": "%Token%",
echo   "SuppressMs": 350,
echo   "AwaitMutationTimeoutMs": 300
echo }
) > "%InstallPath%\config.json"

rem Stop old process
echo [4/6] Stopping old process...
taskkill /F /IM ClipboardClient.exe >nul 2>&1
timeout /t 1 /nobreak >nul

rem Set auto startup
echo [5/6] Setting auto startup...
reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Run" /v "ClipboardListener" /t REG_SZ /d "%InstallPath%\ClipboardClient.exe" /f >nul

rem Start client
echo [6/6] Starting client...
start "" /D "%InstallPath%" "%InstallPath%\ClipboardClient.exe"

echo.
echo ========================================
echo   Installation Complete!
echo   Install Path: %InstallPath%
echo ========================================
echo.
echo Tip: Log file is located at %TEMP%\clipboard-push.log
echo.
