"""
快速测试GPU地址生成
只测试核心功能，不需要完整服务
"""
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# 测试TRON地址生成速度
def test_tron_generation():
    """测试TRON地址生成（使用简单的模拟）"""
    print("=== 测试TRON地址生成速度 ===")
    
    # CPU测试
    print("\n[CPU测试]")
    cpu_count = multiprocessing.cpu_count()
    print(f"CPU核心数: {cpu_count}")
    
    # 模拟地址生成
    attempts = 0
    start_time = time.time()
    duration = 1.0  # 测试1秒
    
    # 简单的性能测试
    while time.time() - start_time < duration:
        # 模拟地址生成计算
        for _ in range(10000):
            attempts += 1
    
    elapsed = time.time() - start_time
    speed = attempts / elapsed
    
    print(f"单线程速度: {speed:,.0f} 次/秒")
    print(f"预计{cpu_count}核并行: {speed * cpu_count:,.0f} 次/秒")
    
    # 计算成功率
    difficulty = 58 ** 4  # TRON地址匹配4位的难度
    success_rate_1s = (speed * cpu_count) / difficulty * 100
    success_rate_1_5s = success_rate_1s * 1.5
    
    print(f"\n难度: {difficulty:,} (58^4)")
    print(f"1秒成功率: {success_rate_1s:.2f}%")
    print(f"1.5秒成功率: {success_rate_1_5s:.2f}%")

# 检查GPU
def check_gpu():
    """检查GPU是否可用"""
    print("\n=== 检查GPU ===")
    
    # 检查NVIDIA GPU
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 检测到NVIDIA GPU:")
            print(result.stdout.strip())
            
            # 简单估算GPU性能
            print("\nGPU性能估算:")
            print("RTX 3060: ~1000万 地址/秒")
            print("RTX 3080: ~3000万 地址/秒")
            print("RTX 4090: ~1亿 地址/秒")
            
            # 计算GPU成功率
            gpu_speed = 30000000  # 假设3000万/秒
            gpu_success_1_5s = (gpu_speed * 1.5) / (58 ** 4) * 100
            print(f"\nGPU 1.5秒成功率: {gpu_success_1_5s:.1f}%")
            
            return True
    except Exception as e:
        print(f"❌ 未检测到NVIDIA GPU: {e}")
    
    return False

# 测试地址模式
def test_address_patterns():
    """测试不同地址的模式提取"""
    print("\n=== 测试地址模式 ===")
    
    test_addresses = {
        "TRON": "TKzxdSv2FZKQrEqkKVgp5DcwEXBEKMg2Ax",
        "ETH": "0x742d35Cc6634C0532925a3b844Bc9e7595f6E321",
        "BTC": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
    }
    
    for coin, address in test_addresses.items():
        if coin == "TRON":
            prefix = address[1:3]  # 跳过T
            suffix = address[-3:]
        elif coin == "ETH":
            prefix = address[2:4]  # 跳过0x
            suffix = address[-3:]
        else:  # BTC
            prefix = address[1:3]  # 跳过1
            suffix = address[-3:]
        
        print(f"{coin}: {address}")
        print(f"  目标模式: {prefix}...{suffix}")

# 性能建议
def performance_recommendations(has_gpu):
    """性能建议"""
    print("\n=== 性能建议 ===")
    
    if has_gpu:
        print("✅ 您有GPU，建议:")
        print("1. 使用GPU工具（profanity2, VanitySearch）")
        print("2. 复杂地址使用GPU生成")
        print("3. 简单地址可以用CPU节省成本")
    else:
        print("⚠️ 未检测到GPU，建议:")
        print("1. 使用CPU多进程优化")
        print("2. 考虑使用云GPU服务")
        print("3. 降低地址复杂度要求")
    
    print("\n成功率参考:")
    print("- 90%+: 建议使用GPU")
    print("- 50-90%: CPU多核可以接受")
    print("- <50%: 考虑降低要求或使用GPU")

if __name__ == "__main__":
    print("Windows GPU地址生成测试\n")
    
    # 运行测试
    test_tron_generation()
    has_gpu = check_gpu()
    test_address_patterns()
    performance_recommendations(has_gpu)
    
    print("\n测试完成！")
