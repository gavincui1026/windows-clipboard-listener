#!/bin/bash

echo "编译测试程序..."
gcc -o test_simple test_simple.cu

if [ $? -eq 0 ]; then
    echo "运行测试..."
    ./test_simple
    rm -f test_simple
else
    echo "编译失败"
fi
