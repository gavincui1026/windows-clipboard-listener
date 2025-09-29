# Windows 客户端简单安装指南

## 准备工作

### 1. 编译客户端并放到服务器
```bash
# 编译客户端
cd client\ClipboardClient
dotnet publish -c Release -r win-x64 --self-contained true -p:PublishSingleFile=true

# 复制到服务器静态目录
copy bin\Release\net8.0-windows\win-x64\publish\ClipboardClient.exe ..\..\server\static\
```

### 2. 启动服务器
```bash
cd server
python main.py
```

服务器启动后会自动：
- 在 `/static/` 路径提供文件下载服务
- 在 `/install.ps1` 路径提供安装脚本

## 安装方法

### 方法1：PowerShell 单行命令（推荐）
```powershell
iwr -useb http://your-server:8001/install.ps1 | iex
```

### 方法2：带参数安装
```powershell
# 指定服务器地址和Token
&([scriptblock]::Create((iwr -useb http://your-server:8001/install.ps1))) -BaseUrl "http://192.168.1.100:8001" -Token "your-token"
```

### 方法3：手动下载安装
```powershell
# 下载安装脚本
Invoke-WebRequest -Uri "http://your-server:8001/install.ps1" -OutFile "install.ps1"

# 执行安装
.\install.ps1 -BaseUrl "http://your-server:8001" -Token "your-token"
```

## 文件位置

安装后的文件位置：
- 程序：`%LOCALAPPDATA%\ClipboardListener\ClipboardClient.exe`
- 配置：`%LOCALAPPDATA%\ClipboardListener\appsettings.json`

## 卸载方法

```powershell
# 停止进程
Stop-Process -Name "ClipboardClient" -Force

# 移除自启动
Remove-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" -Name "ClipboardListener"

# 删除文件
Remove-Item -Path "$env:LOCALAPPDATA\ClipboardListener" -Recurse -Force
```

## 更新客户端

当有新版本时，只需：
1. 将新的 `ClipboardClient.exe` 复制到 `server/static/`
2. 用户重新运行安装命令即可自动更新

## 注意事项

1. 确保服务器防火墙开放了相应端口（默认8001）
2. 生产环境建议使用 HTTPS 和正式的认证 Token
3. 可以通过 Nginx 反向代理提供 HTTPS 支持
