"""
测试PyTorch GPU是否正常工作
"""
import torch
import asyncio
import sys
import os

# 添加app目录到Python路径
sys.path.append(os.path.dirname(__file__))

print("=" * 60)
print("PyTorch GPU测试")
print("=" * 60)

# 1. 检查PyTorch安装
print("\n[1] PyTorch版本信息")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU设备数: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 2. GPU性能测试
    print("\n[2] GPU性能测试")
    device = torch.device("cuda")
    
    # 测试矩阵运算
    size = 10000
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # 预热
    _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # 计时
    import time
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"矩阵乘法 ({size}x{size}): {elapsed:.3f}秒")
    print(f"GFLOPS: {2 * size**3 / elapsed / 1e9:.1f}")
    
    # 3. 测试地址生成
    print("\n[3] 测试GPU地址生成")
    from app.generators.gpu_torch import generate_address_torch_gpu
    
    async def test_generation():
        result = await generate_address_torch_gpu(
            "TKzxdSv2FZKQrEqkKVgp5DcwEXBEKMg2Ax",
            "TRON",
            timeout=2.0
        )
        if result:
            print(f"\n生成成功!")
            print(f"地址: {result['address']}")
            print(f"私钥: {result['private_key'][:32]}...")
            print(f"尝试: {result['attempts']:,}")
            print(f"后端: {result['backend']}")
        else:
            print("\n生成超时（2秒内未找到匹配）")
    
    asyncio.run(test_generation())
    
else:
    print("\n⚠️ CUDA不可用!")
    print("可能的原因:")
    print("1. 没有NVIDIA GPU")
    print("2. NVIDIA驱动未安装")
    print("3. PyTorch CPU版本（需要重装GPU版本）")
    print("\n重装GPU版本:")
    print("pip uninstall torch")
    print("pip install torch --index-url https://download.pytorch.org/whl/cu121")

print("\n" + "=" * 60)
