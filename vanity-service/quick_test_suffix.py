#!/usr/bin/env python3
"""
快速测试后缀匹配功能
"""
import subprocess
import os
import platform

def test_profanity_direct():
    """直接测试profanity命令"""
    
    # 测试生成后5位为AAAAA的地址
    print("测试生成后5位为AAAAA的TRON地址...")
    print("-" * 60)
    
    cmd = [
        "profanity",
        "--matching", "TXXXXXXXXXXXXXXXXXXXXXXXXXXAAAAA",
        "--suffix-count", "5",
        "--quit-count", "1"
    ]
    
    try:
        print("执行命令:", " ".join(cmd))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60秒超时
        )
        
        print("输出:")
        print(result.stdout)
        
        if result.stderr:
            print("错误:")
            print(result.stderr)
            
        print("\n返回码:", result.returncode)
        
    except Exception as e:
        print(f"执行失败: {e}")


if __name__ == "__main__":
    test_profanity_direct()
