#!/usr/bin/env python3
"""
快速测试profanity是否工作
"""
import subprocess
import time

print("=== Profanity快速测试 ===\n")

# 测试1：非常简单的后缀（1个字符）
print("测试1：生成后1位为A的地址（应该立即完成）")
cmd = ["profanity", "--matching", "TXXXXXXXXXXXXXXXXXXA", "--suffix-count", "1", "--quit-count", "1"]
print("命令:", " ".join(cmd))
print("-" * 60)

start = time.time()
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# 等待最多5秒
found = False
while time.time() - start < 5:
    # 检查stdout
    line = proc.stdout.readline()
    if line:
        line = line.strip()
        print(f"[STDOUT] {line}")
        # 检查是否是地址行
        if line and line[0] == 'T' and len(line.split()) >= 2:
            parts = line.split()
            if len(parts[0]) == 34 and len(parts[1]) == 64:
                print(f"\n✅ 找到地址！")
                print(f"地址: {parts[0]}")
                print(f"私钥: {parts[1]}")
                print(f"后1位: {parts[0][-1]}")
                found = True
                proc.terminate()
                break
    
    # 检查进程是否结束
    if proc.poll() is not None:
        break

if not found:
    proc.terminate()
    print("\n❌ 5秒内未找到地址")
    
    # 读取所有输出
    stdout, stderr = proc.communicate()
    if stdout:
        print("\n剩余标准输出:")
        print(stdout)
    if stderr:
        print("\n错误输出:")
        print(stderr)

print(f"\n用时: {time.time() - start:.2f}秒")
print("返回码:", proc.returncode)

# 测试2：检查profanity版本或帮助
print("\n\n测试2：检查profanity是否正确响应")
print("运行: profanity")
print("-" * 60)

proc = subprocess.run(["profanity"], capture_output=True, text=True, timeout=2)
print("返回码:", proc.returncode)
if proc.stdout:
    print("输出（前500字符）:")
    print(proc.stdout[:500])
if proc.stderr:
    print("错误（前500字符）:")
    print(proc.stderr[:500])
