#!/usr/bin/env python3
"""
测试profanity的输出格式
"""
import subprocess
import time

def test_easy_suffix():
    """测试一个容易生成的后缀"""
    print("测试生成后5位为11111的地址（应该很快）...")
    
    cmd = [
        "profanity",
        "--matching", "TXXXXXXXXXXXXXXXXXXXXXXXXXX11111",
        "--suffix-count", "5",
        "--quit-count", "1"
    ]
    
    print("命令:", " ".join(cmd))
    print("开始执行...")
    
    start_time = time.time()
    
    try:
        # 使用Popen来实时读取输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时读取输出
        print("\n--- 标准输出 ---")
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"STDOUT: {output.strip()}")
                # 检查是否是地址行
                if output.strip() and len(output.strip().split()) >= 2:
                    parts = output.strip().split()
                    if parts[0].startswith("T") and len(parts[0]) == 34:
                        print(f"\n找到地址！")
                        print(f"地址: {parts[0]}")
                        print(f"私钥: {parts[1]}")
                        print(f"后5位: {parts[0][-5:]}")
        
        # 读取错误输出
        stderr = process.stderr.read()
        if stderr:
            print("\n--- 错误输出 ---")
            print(f"STDERR: {stderr}")
        
        elapsed = time.time() - start_time
        print(f"\n执行时间: {elapsed:.2f}秒")
        
    except subprocess.TimeoutExpired:
        print("超时！")
    except Exception as e:
        print(f"错误: {e}")


def test_with_different_params():
    """测试不同的参数组合"""
    print("\n\n测试无suffix-count参数...")
    
    cmd = [
        "profanity",
        "--matching", "TXXXXXXXXXXXXXXXXXXXXXXXXXX11111",
        "--quit-count", "1"
    ]
    
    print("命令:", " ".join(cmd))
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print("返回码:", result.returncode)
        print("输出:", result.stdout)
        if result.stderr:
            print("错误:", result.stderr)
            
    except subprocess.TimeoutExpired:
        print("超时！")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    print("=== Profanity输出格式测试 ===\n")
    test_easy_suffix()
    test_with_different_params()
