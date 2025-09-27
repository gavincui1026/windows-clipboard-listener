"""
GPU工具包装器
集成外部GPU工具（profanity2, VanitySearch等）
"""
import os
import subprocess
import asyncio
import json
import tempfile
from typing import Dict, Optional


# 检查GPU是否可用
def check_gpu_available():
    """检测GPU是否可用（跨平台）"""
    # 方法1: 检查NVIDIA-SMI（跨平台）
    try:
        import subprocess
        # Windows可能需要shell=True
        nvidia_cmd = ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader']
        result = subprocess.run(nvidia_cmd, 
                              capture_output=True, 
                              text=True,
                              shell=(os.name == 'nt'))  # Windows需要shell
        if result.returncode == 0 and result.stdout.strip():
            return True
    except Exception as e:
        pass
    
    # 方法2: 检查pycuda（可选）
    try:
        import pycuda.driver as cuda
        cuda.init()
        if cuda.Device.count() > 0:
            return True
    except:
        pass
    
    # 方法3: 检查GPU工具是否存在（跨平台）
    gpu_tools_path = os.getenv("GPU_TOOLS_PATH", "./gpu_tools")
    
    # 根据平台检查不同的文件
    if os.name == 'nt':  # Windows
        profanity2 = os.path.join(gpu_tools_path, "profanity2.exe")
        vanitysearch = os.path.join(gpu_tools_path, "VanitySearch.exe")
    else:  # Linux/Unix
        profanity2 = os.path.join(gpu_tools_path, "profanity2")
        vanitysearch = os.path.join(gpu_tools_path, "VanitySearch")
    
    # 如果找到任何GPU工具，认为GPU可用
    if os.path.exists(profanity2) or os.path.exists(vanitysearch):
        return True
    
    return False

GPU_AVAILABLE = check_gpu_available()

# GPU工具路径配置
GPU_TOOLS_PATH = os.getenv("GPU_TOOLS_PATH", "./gpu_tools")

# Windows下需要.exe扩展名
if os.name == 'nt':  # Windows
    PROFANITY2_PATH = os.path.join(GPU_TOOLS_PATH, "profanity2.exe")
    VANITYSEARCH_PATH = os.path.join(GPU_TOOLS_PATH, "VanitySearch.exe")
else:
    PROFANITY2_PATH = os.path.join(GPU_TOOLS_PATH, "profanity2")
    VANITYSEARCH_PATH = os.path.join(GPU_TOOLS_PATH, "VanitySearch")


async def generate_with_gpu(original_address: str, address_type: str) -> Optional[Dict]:
    """
    使用GPU生成地址
    """
    if not GPU_AVAILABLE:
        return None
    
    # 根据地址类型选择工具
    if address_type in ['ETH', 'BNB']:
        return await generate_eth_gpu(original_address)
    elif address_type.startswith('BTC'):
        return await generate_btc_gpu(original_address, address_type)
    elif address_type == 'TRON':
        return await generate_tron_gpu(original_address)
    else:
        return None


async def generate_eth_gpu(address: str) -> Optional[Dict]:
    """使用profanity2生成ETH/BNB地址"""
    if not os.path.exists(PROFANITY2_PATH):
        print(f"profanity2未找到: {PROFANITY2_PATH}")
        return None
    
    # 提取模式
    prefix = address[2:4]  # 跳过0x
    suffix = address[-3:]
    
    # 构建命令
    cmd = [
        PROFANITY2_PATH,
        "--matching", f"{prefix}",  # 前缀匹配
        "--suffix", suffix,         # 后缀匹配
        "--output", "json",         # JSON输出
        "--limit", "1"              # 只生成一个
    ]
    
    try:
        # 执行命令
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=2.0
        )
        
        if process.returncode == 0:
            # 解析结果
            result = json.loads(stdout.decode())
            return {
                'address': result['address'],
                'private_key': result['privateKey'],
                'type': 'ETH',
                'attempts': result.get('attempts', 0)
            }
    except Exception as e:
        print(f"profanity2执行失败: {e}")
    
    return None


async def generate_btc_gpu(address: str, address_type: str) -> Optional[Dict]:
    """使用VanitySearch生成BTC地址"""
    if not os.path.exists(VANITYSEARCH_PATH):
        print(f"VanitySearch未找到: {VANITYSEARCH_PATH}")
        return None
    
    # 提取模式
    if address_type == 'BTC_P2PKH':
        prefix = "1" + address[1:3]
    elif address_type == 'BTC_P2SH':
        prefix = "3" + address[1:3]
    else:  # Bech32
        prefix = "bc1" + address[3:5]
    
    suffix = address[-3:]
    
    # 构建命令
    cmd = [
        VANITYSEARCH_PATH,
        "-gpu",                    # 使用GPU
        "-o", "result.txt",        # 输出文件
        f"{prefix}*{suffix}"       # 模式
    ]
    
    try:
        # 使用临时文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp_path = tmp.name
        
        cmd[3] = tmp_path  # 更新输出路径
        
        # 执行命令
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=2.0
        )
        
        if process.returncode == 0 and os.path.exists(tmp_path):
            # 读取结果
            with open(tmp_path, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    address = lines[0].strip()
                    private_key = lines[1].strip()
                    return {
                        'address': address,
                        'private_key': private_key,
                        'type': address_type,
                        'attempts': 1000000  # VanitySearch不报告尝试次数
                    }
        
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    except Exception as e:
        print(f"VanitySearch执行失败: {e}")
    
    return None


async def generate_tron_gpu(address: str) -> Optional[Dict]:
    """使用自定义CUDA程序生成TRON地址"""
    # 检查是否有编译好的CUDA程序
    tron_gpu_path = os.path.join(GPU_TOOLS_PATH, "tron_vanity_gpu")
    
    if not os.path.exists(tron_gpu_path):
        print(f"TRON GPU程序未找到: {tron_gpu_path}")
        return None
    
    # 构建命令
    cmd = [
        tron_gpu_path,
        "--address", address,
        "--timeout", "1.5"
    ]
    
    try:
        # 执行命令
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=2.0
        )
        
        if process.returncode == 0:
            # 解析输出
            lines = stdout.decode().strip().split('\n')
            if len(lines) >= 3:
                return {
                    'address': lines[0],
                    'private_key': lines[1],
                    'type': 'TRON',
                    'attempts': int(lines[2])
                }
    except Exception as e:
        print(f"TRON GPU执行失败: {e}")
    
    return None


# GPU设置指南
GPU_SETUP_GUIDE = """
GPU工具设置指南：

1. profanity2 (ETH/BNB)
   - 下载: https://github.com/1inch/profanity2/releases
   - 放置到: ./gpu_tools/profanity2

2. VanitySearch (BTC)
   - 下载: https://github.com/JeanLucPons/VanitySearch/releases
   - 放置到: ./gpu_tools/VanitySearch

3. TRON GPU (自定义)
   - 编译CUDA代码: cd gpu && nvcc -O3 -o tron_vanity_gpu tron_vanity.cu
   - 放置到: ./gpu_tools/tron_vanity_gpu

环境变量：
- GPU_TOOLS_PATH: GPU工具目录路径（默认./gpu_tools）
"""
