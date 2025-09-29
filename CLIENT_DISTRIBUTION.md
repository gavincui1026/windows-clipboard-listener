# 客户端程序分发方案

由于编译后的 `ClipboardClient.exe` 文件超过 100MB，无法直接存储在 GitHub 仓库中。以下是几种分发方案：

## 方案1：使用 GitHub Releases（推荐）

1. **创建 Release**：
   ```bash
   # 编译客户端
   cd client\ClipboardClient
   dotnet publish -c Release -r win-x64 --self-contained true -p:PublishSingleFile=true
   
   # 压缩exe文件（可选，减小体积）
   7z a -mx9 ClipboardClient.7z bin\Release\net8.0-windows\win-x64\publish\ClipboardClient.exe
   ```

2. **在 GitHub 上发布**：
   - 访问 https://github.com/gavincui1026/windows-clipboard-listener/releases
   - 点击 "Create a new release"
   - 上传 `ClipboardClient.exe` 或压缩包
   - 发布 Release

3. **修改安装脚本**：
   ```powershell
   # 修改 install-simple.ps1 中的下载地址
   $downloadUrl = "https://github.com/gavincui1026/windows-clipboard-listener/releases/latest/download/ClipboardClient.exe"
   ```

## 方案2：使用云存储服务

将客户端程序上传到：
- **阿里云 OSS**
- **腾讯云 COS**
- **七牛云**
- **CloudFlare R2**

然后修改安装脚本中的下载地址。

## 方案3：部署时编译

在 `server/static/.gitkeep` 中添加说明：
```
# 部署时需要手动放置 ClipboardClient.exe
# 编译命令：
# cd client\ClipboardClient
# dotnet publish -c Release -r win-x64 --self-contained true -p:PublishSingleFile=true
# copy bin\Release\net8.0-windows\win-x64\publish\ClipboardClient.exe ..\..\server\static\
```

## 方案4：减小文件体积

使用以下方式减小编译后的文件大小：

1. **修改项目文件** `ClipboardClient.csproj`：
   ```xml
   <PropertyGroup>
     <PublishTrimmed>true</PublishTrimmed>
     <PublishReadyToRun>true</PublishReadyToRun>
     <PublishSingleFile>true</PublishSingleFile>
     <SelfContained>true</SelfContained>
     <EnableCompressionInSingleFile>true</EnableCompressionInSingleFile>
   </PropertyGroup>
   ```

2. **使用 UPX 压缩**：
   ```bash
   # 下载 UPX: https://github.com/upx/upx/releases
   upx --best ClipboardClient.exe
   ```

## 方案5：动态下载方式

修改安装脚本，支持多个下载源：

```powershell
# 在 install-simple.ps1 中
$downloadUrls = @(
    "$BaseUrl/static/ClipboardClient.exe",  # 优先从服务器下载
    "https://github.com/gavincui1026/windows-clipboard-listener/releases/latest/download/ClipboardClient.exe",  # GitHub Release
    "https://your-cdn.com/ClipboardClient.exe"  # CDN备用地址
)

foreach ($url in $downloadUrls) {
    try {
        Invoke-WebRequest -Uri $url -OutFile "$InstallPath\ClipboardClient.exe" -UseBasicParsing
        Write-Host "从 $url 下载成功" -ForegroundColor Green
        break
    } catch {
        Write-Host "从 $url 下载失败，尝试下一个..." -ForegroundColor Yellow
    }
}
```

## 推荐做法

1. 使用 GitHub Releases 发布正式版本
2. 在服务器上部署时手动放置客户端文件
3. 安装脚本支持多个下载源，提高可用性
4. 考虑使用 CDN 加速国内用户下载
