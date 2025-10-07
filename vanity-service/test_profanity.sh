#!/bin/bash
# 测试profanity-tron后缀匹配功能

echo "=== 测试profanity-tron后缀匹配 ==="
echo

# 进入profanity-tron目录
cd profanity-tron

# 测试生成后5位为AAAAA的地址
echo "测试1: 生成后5位为AAAAA的地址"
./profanity.x64 --matching TXXXXXXXXXXXXXXXXXXXXXXXXXXAAAAA --suffix-count 5 --quit-count 1
echo

# 测试生成后5位为12345的地址
echo "测试2: 生成后5位为12345的地址"
./profanity.x64 --matching TXXXXXXXXXXXXXXXXXXXXXXXXXX12345 --suffix-count 5 --quit-count 1
echo

# 测试生成后5位为ekjF5的地址（模拟实际需求）
echo "测试3: 生成后5位为ekjF5的地址"
./profanity.x64 --matching TXXXXXXXXXXXXXXXXXXXXXXXXXekjF5 --suffix-count 5 --quit-count 1
echo

cd ..

echo "=== 测试Python集成 ==="
python test_suffix_generation.py
