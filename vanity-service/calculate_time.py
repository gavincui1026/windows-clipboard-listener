"""
计算生成特定模式地址的理论时间
"""

def calculate_generation_time(pattern: str, speed: int = 1_000_000):
    """
    计算生成时间
    pattern: 地址模式，如 "TKz...Ax"
    speed: 每秒生成速度
    """
    # 提取前缀和后缀
    if '...' in pattern:
        parts = pattern.split('...')
        prefix = parts[0][1:]  # 去掉T
        suffix = parts[1] if len(parts) > 1 else ""
    else:
        prefix = pattern[1:] if pattern.startswith('T') else pattern
        suffix = ""
    
    # 计算需要匹配的字符数
    match_chars = len(prefix) + len(suffix)
    
    # Base58字符集大小
    base58_size = 58
    
    # 计算概率和期望尝试次数
    probability = 1 / (base58_size ** match_chars)
    expected_attempts = 1 / probability
    
    # 计算时间
    expected_time_seconds = expected_attempts / speed
    
    # 成功率计算
    success_rates = {}
    for seconds in [1, 2, 10, 60, 300, 3600]:
        attempts = speed * seconds
        success_rate = 1 - ((1 - probability) ** attempts)
        success_rates[seconds] = success_rate * 100
    
    print(f"📊 地址生成时间分析")
    print(f"=" * 50)
    print(f"目标模式: {pattern}")
    print(f"匹配字符: {match_chars}个 (前缀:{len(prefix)} + 后缀:{len(suffix)})")
    print(f"生成速度: {speed:,}/秒")
    print(f"\n概率计算:")
    print(f"  单次概率: 1/{expected_attempts:,.0f}")
    print(f"  期望尝试: {expected_attempts:,.0f}次")
    print(f"\n时间预估:")
    
    if expected_time_seconds < 60:
        print(f"  平均时间: {expected_time_seconds:.1f}秒")
    elif expected_time_seconds < 3600:
        print(f"  平均时间: {expected_time_seconds/60:.1f}分钟")
    elif expected_time_seconds < 86400:
        print(f"  平均时间: {expected_time_seconds/3600:.1f}小时")
    else:
        print(f"  平均时间: {expected_time_seconds/86400:.1f}天")
    
    print(f"\n成功率预测:")
    print(f"  1秒内: {success_rates[1]:.1f}%")
    print(f"  2秒内: {success_rates[2]:.1f}%")
    print(f"  10秒内: {success_rates[10]:.1f}%")
    print(f"  1分钟内: {success_rates[60]:.1f}%")
    print(f"  5分钟内: {success_rates[300]:.1f}%")
    print(f"  1小时内: {success_rates[3600]:.1f}%")
    
    print(f"\n💡 建议:")
    if match_chars <= 4:
        print(f"  ✅ 难度适中，通常能在1分钟内生成")
    elif match_chars == 5:
        print(f"  ⚠️ 难度较高，可能需要5-30分钟")
    elif match_chars == 6:
        print(f"  ❌ 难度很高，可能需要数小时")
    else:
        print(f"  ⛔ 极高难度，建议减少匹配字符")
    
    return {
        'match_chars': match_chars,
        'probability': probability,
        'expected_attempts': expected_attempts,
        'expected_time': expected_time_seconds,
        'success_rates': success_rates
    }


if __name__ == "__main__":
    print("🔍 测试不同模式的生成时间\n")
    
    # 你的24核CPU，优化后约100万/秒
    cpu_speed = 1_000_000
    
    # 测试不同难度
    patterns = [
        "TKz",           # 2个字符
        "TKzx",          # 3个字符
        "TKzxd",         # 4个字符
        "TKz...Ax",      # 5个字符 (3+2)
        "TKzx...Ax",     # 6个字符 (4+2)
        "T11...F5N",     # 6个字符 (2+3)
        "TKzxdS...2Ax",  # 8个字符 (6+2)
    ]
    
    for pattern in patterns:
        calculate_generation_time(pattern, cpu_speed)
        print("\n" + "="*50 + "\n")
