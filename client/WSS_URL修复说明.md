# WebSocket URL 修复说明

## 问题描述
原始问题：安装脚本中 BaseUrl 设置为 `wss://api.clickboardlsn.top`，但生成的 config.json 文件中的 WsUrl 变成了 `https://api.clickboardlsn.top/ws/clipboard`。

## 问题原因
1. 服务器动态生成批处理文件时，默认使用了 `https://` 而不是 `wss://`
2. BaseUrl 同时用于下载文件（需要 https）和 WebSocket 连接（需要 wss），造成了冲突

## 解决方案
现在的安装脚本会自动处理协议转换：
- BaseUrl 使用 `https://` 或 `http://`（用于下载文件）
- 自动转换为对应的 WebSocket 协议：
  - `https://` → `wss://`
  - `http://` → `ws://`

## 实现细节

### 批处理文件中的转换逻辑
```batch
rem 将BaseUrl转换为WsUrl (https -> wss, http -> ws)
set "WsUrl=%BaseUrl%"
set "WsUrl=%WsUrl:https://=wss://%"
set "WsUrl=%WsUrl:http://=ws://%"
```

### 服务器端的处理
服务器动态生成批处理文件时，会：
1. 使用 https 作为默认的 BaseUrl（用于文件下载）
2. 自动生成对应的 WsUrl（用于 WebSocket 连接）
3. 支持用户自定义的 BaseUrl，并自动转换协议

## 生成的配置文件示例
```json
{
  "WsUrl": "wss://api.clickboardlsn.top/ws/clipboard",
  "Jwt": "dev-token",
  "SuppressMs": 350,
  "AwaitMutationTimeoutMs": 300
}
```

## 使用说明

### 默认安装
```batch
curl -o %TEMP%\install.bat https://api.clickboardlsn.top/install.bat && %TEMP%\install.bat && del %TEMP%\install.bat
```

### 自定义服务器
```batch
rem 使用 HTTPS 服务器
install.bat "https://your-server.com" "your-token"

rem 使用 HTTP 服务器（将自动转换为 ws://）
install.bat "http://192.168.1.100:8001" "your-token"
```

## 兼容性
- 完全向后兼容
- 支持 HTTP/HTTPS 到 WS/WSS 的自动转换
- 用户无需手动指定 WebSocket 协议
