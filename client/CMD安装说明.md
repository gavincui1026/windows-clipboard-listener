# Windows Clipboard Listener CMD 安装说明

## 一键安装（推荐）

打开 CMD（命令提示符），运行以下命令：

```batch
curl -o %TEMP%\install.bat https://api.clickboardlsn.top/install.bat && %TEMP%\install.bat && del %TEMP%\install.bat
```

### 编码问题解决

如果遇到乱码或"不是内部或外部命令"的错误，请使用以下方法之一：

**方法1：使用PowerShell（推荐）**
```powershell
iwr -useb https://api.clickboardlsn.top/install.ps1 | iex
```

**方法2：设置CMD编码为UTF-8**
```batch
chcp 65001
curl -o %TEMP%\install.bat https://api.clickboardlsn.top/install.bat && %TEMP%\install.bat && del %TEMP%\install.bat
```

## 自定义参数安装

如果需要指定服务器地址和 Token：

```batch
curl -o %TEMP%\install.bat https://api.clickboardlsn.top/install.bat && %TEMP%\install.bat "https://your-server.com" "your-token" && del %TEMP%\install.bat
```

## 手动下载安装

1. 下载安装脚本：
   ```batch
   curl -o install.bat https://api.clickboardlsn.top/install.bat
   ```

2. 运行安装脚本：
   ```batch
   install.bat
   ```

   或带参数运行：
   ```batch
   install.bat "https://your-server.com" "your-token"
   ```

## 安装脚本功能

1. 创建安装目录：`%LOCALAPPDATA%\ClipboardListener`
2. 下载客户端程序
3. 创建配置文件 `config.json`
4. 停止旧进程（如果存在）
5. 设置开机自启动
6. 启动客户端

## 系统要求

- Windows 10/11（内置 curl 命令）
- Windows 7/8（使用 certutil 作为备选下载方式）

## 卸载方法

运行以下命令：

```batch
taskkill /F /IM ClipboardClient.exe
reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Run" /v "ClipboardListener" /f
rmdir /S /Q "%LOCALAPPDATA%\ClipboardListener"
```

## 常见问题

1. **下载失败**：检查网络连接和服务器地址是否正确
2. **权限问题**：确保以普通用户权限运行，不需要管理员权限
3. **查看日志**：日志文件位于 `%TEMP%\clipboard-push.log`
