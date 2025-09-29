#!/usr/bin/env python3
"""
调试前缀模式匹配
"""

# 从您的截图中的数据
original_pattern = "T3jWrs"
generated_address = "T3jWrsnD51hQQSsSr0ZJebmgCtkMHLeN"

print("=" * 50)
print("前缀匹配验证")
print("=" * 50)

print(f"\n目标模式: {original_pattern} (长度: {len(original_pattern)})")
print(f"生成地址: {generated_address}")
print(f"地址前缀: {generated_address[:len(original_pattern)]}")

# 验证匹配
if generated_address.startswith(original_pattern):
    print(f"\n✓ 匹配成功！地址确实以 '{original_pattern}' 开头")
else:
    print(f"\n✗ 匹配失败！")

# 计算难度
print(f"\n理论难度估算:")
print(f"- TRON地址使用Base58编码")
print(f"- 需要匹配 T + 5位 = 6位")
print(f"- 理论组合数: 58^5 = {58**5:,}")

# 分析为什么这么快
print(f"\n可能的原因:")
print("1. GPU性能强大（oclvanitygen++ 使用GPU加速）")
print("2. 运气好，刚好很快找到匹配")
print("3. OpenCL优化效果好")

# 测试其他地址类型的模式
print("\n" + "=" * 50)
print("其他地址类型的模式构建:")

# ETH
eth_addr = "0x742d35Cc6634C0532925a3b844Bc9e7595f6E321"
print(f"\nETH地址: {eth_addr}")
print(f"匹配模式: {eth_addr[:7]} (0x + 5位十六进制)")
print(f"理论组合数: 16^5 = {16**5:,}")

# BTC
btc_addr = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
print(f"\nBTC地址: {btc_addr}")
print(f"匹配模式: {btc_addr[:6]} (1 + 5位Base58)")
print(f"理论组合数: 58^5 = {58**5:,}")
