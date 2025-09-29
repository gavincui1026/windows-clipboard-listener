# 修复CMD安装脚本编码问题

## 问题描述
使用CMD执行安装命令时出现乱码错误：
```
'�户端失败！' is not recognized as an internal or external command
```

这是因为批处理文件使用UTF-8编码，而Windows CMD默认使用GBK编码导致的。

## 解决方案

### 方案1：使用PowerShell安装（推荐）
PowerShell对编码支持更好，推荐使用PowerShell安装：

```powershell
iwr -useb https://api.clickboardlsn.top/install.ps1 | iex
```

### 方案2：使用修复后的CMD命令
服务器已经更新为提供英文版本的批处理文件，避免编码问题：

```batch
curl -o %TEMP%\install.bat https://api.clickboardlsn.top/install.bat && %TEMP%\install.bat && del %TEMP%\install.bat
```

### 方案3：手动设置CMD编码
在执行安装前，先设置CMD的编码为UTF-8：

```batch
chcp 65001
curl -o %TEMP%\install.bat https://api.clickboardlsn.top/install.bat && %TEMP%\install.bat && del %TEMP%\install.bat
```

### 方案4：下载后手动执行
1. 先下载安装脚本：
   ```batch
   curl -o install.bat https://api.clickboardlsn.top/install.bat
   ```

2. 用记事本打开 install.bat，另存为时选择 ANSI 编码

3. 执行转换后的脚本：
   ```batch
   install.bat
   ```

## 服务器端修复

服务器已经进行了以下修复：
1. 创建了英文版本的批处理文件 `install-en.bat`
2. 修改服务器动态生成纯ASCII批处理内容
3. 设置响应编码为 ASCII
4. 去除了安装完成后的交互提示，自动完成安装

## 测试方法

重启服务器后，测试安装命令是否正常工作：

```batch
curl -o %TEMP%\test.bat https://api.clickboardlsn.top/install.bat && type %TEMP%\test.bat
```

查看下载的文件内容是否为英文版本。
