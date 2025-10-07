#!/usr/bin/env python3
"""
直接运行profanity命令并观察输出
"""
import subprocess
import threading
import time

def read_output(proc, name):
    """读取进程输出"""
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        print(f"[{name}] {line.decode().strip()}")

def read_error(proc, name):
    """读取错误输出"""
    while True:
        line = proc.stderr.readline()
        if not line:
            break
        print(f"[{name}-ERR] {line.decode().strip()}")

# 测试1：生成简单的后缀
print("=== 测试1：生成后5位为11111的地址 ===")
cmd = ["profanity", "--matching", "TXXXXXXXXXXXXXXXXXXXXXXXXXX11111", "--suffix-count", "5", "--quit-count", "1"]
print("命令:", " ".join(cmd))

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# 创建线程读取输出
t1 = threading.Thread(target=read_output, args=(proc, "STDOUT"))
t2 = threading.Thread(target=read_error, args=(proc, "STDERR"))
t1.start()
t2.start()

# 等待最多30秒
start_time = time.time()
while proc.poll() is None and time.time() - start_time < 30:
    time.sleep(0.1)

if proc.poll() is None:
    print("超时，终止进程...")
    proc.terminate()
    
t1.join()
t2.join()

print(f"\n返回码: {proc.returncode}")
print(f"用时: {time.time() - start_time:.2f}秒")

# 测试2：使用--output参数
print("\n\n=== 测试2：使用--output参数 ===")
output_file = "test_result.txt"

# 删除旧文件
import os
if os.path.exists(output_file):
    os.remove(output_file)

cmd = ["profanity", "--matching", "TXXXXXXXXXXXXXXXXXXXXXXXXXX22222", "--suffix-count", "5", "--quit-count", "1", "--output", output_file]
print("命令:", " ".join(cmd))

result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
print(f"返回码: {result.returncode}")

if os.path.exists(output_file):
    print(f"\n输出文件内容:")
    with open(output_file, 'r') as f:
        print(f.read())
    os.remove(output_file)
else:
    print("输出文件不存在")

print("\n标准输出:")
print(result.stdout)
print("\n错误输出:")
print(result.stderr)
