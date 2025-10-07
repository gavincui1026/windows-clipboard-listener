#!/usr/bin/env python3
"""
查看profanity的帮助信息
"""
import subprocess

print("=== Profanity帮助信息 ===\n")

# 测试 --help
cmd = ["profanity", "--help"]
print("执行:", " ".join(cmd))
print("-" * 60)

try:
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("返回码:", result.returncode)
    print("\n标准输出:")
    print(result.stdout)
    if result.stderr:
        print("\n错误输出:")
        print(result.stderr)
except Exception as e:
    print(f"错误: {e}")

# 测试不带参数
print("\n\n=== 不带参数运行 ===")
cmd = ["profanity"]
print("执行:", " ".join(cmd))
print("-" * 60)

try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
    print("返回码:", result.returncode)
    print("\n标准输出:")
    print(result.stdout)
    if result.stderr:
        print("\n错误输出:")
        print(result.stderr)
except subprocess.TimeoutExpired:
    print("超时（2秒）")
except Exception as e:
    print(f"错误: {e}")
