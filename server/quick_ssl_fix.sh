#!/bin/bash
# 快速修复SSL证书验证问题

echo "=== SSL证书问题快速修复 ==="
echo

# 检查是否存在.env文件
ENV_FILE="server/.env"
if [ ! -f "$ENV_FILE" ]; then
    # 如果在server目录下运行
    if [ -f ".env" ]; then
        ENV_FILE=".env"
    elif [ -f "env.example" ]; then
        cp env.example .env
        ENV_FILE=".env"
    else
        echo "错误：找不到.env文件"
        exit 1
    fi
fi

# 备份原文件
cp "$ENV_FILE" "$ENV_FILE.ssl_backup"
echo "已备份配置文件到: $ENV_FILE.ssl_backup"

# 检查是否已有VERIFY_SSL配置
if grep -q "^VERIFY_SSL=" "$ENV_FILE"; then
    # 更新现有配置
    sed -i 's/^VERIFY_SSL=.*/VERIFY_SSL=false/' "$ENV_FILE"
    echo "已更新VERIFY_SSL配置为false"
else
    # 添加新配置
    echo "" >> "$ENV_FILE"
    echo "# SSL验证配置（临时禁用以解决证书问题）" >> "$ENV_FILE"
    echo "VERIFY_SSL=false" >> "$ENV_FILE"
    echo "已添加VERIFY_SSL=false配置"
fi

echo
echo "修复完成！"
echo
echo "注意事项："
echo "1. 这是临时解决方案，仅用于测试环境"
echo "2. 生产环境建议使用有效的SSL证书"
echo "3. 或使用本地vanity服务: VANITY_SERVICE_URL=http://localhost:8002"
echo
echo "请重启服务以应用更改："
echo "  docker-compose restart api"
echo "  或"
echo "  systemctl restart your-api-service"
