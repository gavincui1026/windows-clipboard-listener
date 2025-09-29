# Windows Clipboard Listener 简单安装脚本
# 使用方法: iwr -useb https://your-server.com/install.ps1 | iex

param(
    [string]$BaseUrl = "wss://api.clickboardlsn.top",
    [string]$Token = "dev-token"
)

$InstallPath = "$env:LOCALAPPDATA\ClipboardListener"

Write-Host "正在安装 Windows Clipboard Listener..." -ForegroundColor Cyan

# 创建安装目录
New-Item -ItemType Directory -Path $InstallPath -Force | Out-Null

# 下载客户端
Write-Host "下载客户端程序..." -ForegroundColor Yellow
Invoke-WebRequest -Uri "$BaseUrl/static/ClipboardClient.exe" -OutFile "$InstallPath\ClipboardClient.exe" -UseBasicParsing

# 创建配置文件
Write-Host "创建配置文件..." -ForegroundColor Yellow
$config = @{
    WsUrl = "$BaseUrl/ws/clipboard"
    Jwt = $Token
    SuppressMs = 350
    AwaitMutationTimeoutMs = 300
}
$config | ConvertTo-Json | Set-Content -Path "$InstallPath\config.json" -Encoding UTF8

# 停止旧进程
Get-Process -Name "ClipboardClient" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

# 设置开机自启动
Write-Host "设置开机自启动..." -ForegroundColor Yellow
$regPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
Set-ItemProperty -Path $regPath -Name "ClipboardListener" -Value "$InstallPath\ClipboardClient.exe" -Force

# 启动客户端
Write-Host "启动客户端..." -ForegroundColor Yellow
Start-Process -FilePath "$InstallPath\ClipboardClient.exe" -WorkingDirectory $InstallPath -WindowStyle Hidden

Write-Host "✓ 安装完成！" -ForegroundColor Green
Write-Host "安装路径: $InstallPath" -ForegroundColor Gray
