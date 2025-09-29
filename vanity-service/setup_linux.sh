#!/bin/bash
# Linux 环境下的 vanity-service 设置脚本

echo "=========================================="
echo "Vanity Service Linux 环境设置"
echo "=========================================="

# 1. 检查并安装 Python 依赖
echo -e "\n[1] 安装 Python 依赖..."
pip install -r requirements.txt

# 2. 构建 vanitygen-plusplus
echo -e "\n[2] 构建 vanitygen-plusplus..."
chmod +x build_vanitygen.sh
./build_vanitygen.sh

# 3. 检查可执行文件
echo -e "\n[3] 检查可执行文件..."
python3 -c "
from app.utils.vanitygen_plusplus import is_vpp_available, _find_all_exes
print('VPP Available:', is_vpp_available())
print('Found executables:')
for exe in _find_all_exes():
    print(' -', exe)
"

# 4. 创建 .env 文件（如果不存在）
if [ ! -f .env ]; then
    echo -e "\n[4] 创建 .env 配置文件..."
    cp env.example .env
    echo "请编辑 .env 文件设置必要的配置"
fi

# 5. 设置环境变量
echo -e "\n[5] 环境变量设置..."
echo "设置更长的超时时间（建议 10-30 秒）："
echo "export DEFAULT_TIMEOUT=15"

# 6. 启用调试（可选）
echo -e "\n[6] 调试选项："
echo "启用调试日志：export VPP_DEBUG=1"
echo "指定 GPU 设备：export VPP_PLATFORM=0 VPP_DEVICE=0"

echo -e "\n=========================================="
echo "设置完成！"
echo ""
echo "启动服务："
echo "  python main.py"
echo ""
echo "或使用后台运行："
echo "  nohup python main.py > vanity.log 2>&1 &"
echo "=========================================="
