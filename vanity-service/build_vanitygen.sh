#!/bin/bash
# 在 Linux 服务器上构建 vanitygen-plusplus

echo "构建 vanitygen-plusplus..."

# 进入 vanitygen-plusplus 目录
cd vanitygen-plusplus

# 安装依赖（Ubuntu/Debian）
if command -v apt-get &> /dev/null; then
    echo "安装 Ubuntu/Debian 依赖..."
    sudo apt-get update
    sudo apt-get install -y build-essential libssl-dev libpcre3-dev libcurl4-openssl-dev ocl-icd-opencl-dev
fi

# 安装依赖（CentOS/RHEL）  
if command -v yum &> /dev/null; then
    echo "安装 CentOS/RHEL 依赖..."
    sudo yum install -y gcc make openssl-devel pcre-devel libcurl-devel ocl-icd-devel
fi

# 清理旧的构建文件
make clean 2>/dev/null || true

# 构建所有版本
echo "开始构建..."
make all

# 检查构建结果
echo -e "\n构建结果："
if [ -f "vanitygen++" ]; then
    echo "✓ vanitygen++ (CPU版本) 构建成功"
    chmod +x vanitygen++
fi

if [ -f "oclvanitygen++" ]; then
    echo "✓ oclvanitygen++ (GPU版本) 构建成功"
    chmod +x oclvanitygen++
fi

if [ -f "keyconv" ]; then
    echo "✓ keyconv 构建成功"
    chmod +x keyconv
fi

# 如果没有 GPU 版本，尝试构建不带 OpenCL 的版本
if [ ! -f "oclvanitygen++" ]; then
    echo -e "\n注意：GPU 版本构建失败，尝试构建纯 CPU 版本..."
    make vanitygen++ keyconv
fi

# 创建 bin 目录并复制可执行文件
mkdir -p bin
cp -f vanitygen++ bin/ 2>/dev/null || true
cp -f oclvanitygen++ bin/ 2>/dev/null || true
cp -f keyconv bin/ 2>/dev/null || true

echo -e "\n构建完成！"
echo "可执行文件位置："
ls -la bin/

cd ..
