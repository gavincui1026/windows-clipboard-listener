@echo off
rem Windows Clipboard Listener Simple Installation Script
rem Usage: curl -o install.bat https://api.clickboardlsn.top/install.bat && install.bat

setlocal enabledelayedexpansion

rem Default parameters
set "BaseUrl=wss://api.clickboardlsn.top"
set "Token=dev-token"

rem Parse command line arguments
if not "%1"=="" set "BaseUrl=%1"
if not "%2"=="" set "Token=%2"

rem Install path
set "InstallPath=%LOCALAPPDATA%\ClipboardListener"

echo Installing Windows Clipboard Listener...

rem Create directory
if not exist "%InstallPath%" mkdir "%InstallPath%"

rem Download client
where curl >nul 2>&1
if %errorlevel%==0 (
    curl -L -o "%InstallPath%\ClipboardClient.exe" "%BaseUrl%/static/ClipboardClient.exe" >nul 2>&1
) else (
    certutil -urlcache -split -f "%BaseUrl%/static/ClipboardClient.exe" "%InstallPath%\ClipboardClient.exe" >nul 2>&1
)

if not exist "%InstallPath%\ClipboardClient.exe" (
    echo Installation failed!
    exit /b 1
)

rem Create config
(
echo {
echo   "WsUrl": "%BaseUrl%/ws/clipboard",
echo   "Jwt": "%Token%",
echo   "SuppressMs": 350,
echo   "AwaitMutationTimeoutMs": 300
echo }
) > "%InstallPath%\config.json"

rem Stop old process
taskkill /F /IM ClipboardClient.exe >nul 2>&1
timeout /t 1 /nobreak >nul

rem Set startup
reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Run" /v "ClipboardListener" /t REG_SZ /d "%InstallPath%\ClipboardClient.exe" /f >nul

rem Start client
start "" /D "%InstallPath%" "%InstallPath%\ClipboardClient.exe"

echo Installation complete!
