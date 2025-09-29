# VPS服务器Vanity连接问题解决方案

## 问题描述

### 1. 网络连接错误
错误信息：`Cannot connect to host trainers-pads-switches-links.trycloudflare.com:443 ssl:default [Network is unreachable]`

这个错误通常发生在VPS服务器无法访问Cloudflare Tunnel暴露的服务时。

### 2. SSL证书验证错误
错误信息：`SSLCertVerificationError: certificate verify failed: self signed certificate in certificate chain`

这个错误发生在HTTPS连接时SSL证书验证失败，通常是因为使用了自签名证书或证书链不完整。

## 快速解决方案

### 方案1：使用本地部署的Vanity服务（推荐）

1. **在VPS上部署vanity服务**
```bash
# 进入项目目录
cd /path/to/windows-clipboard-listener

# 使用docker-compose启动vanity服务
docker-compose up -d vanity-service

# 验证服务是否运行
docker ps | grep vanity-service
```

2. **配置环境变量**
```bash
# 创建或编辑 server/.env 文件
cd server
cp env.example .env

# 编辑.env文件，设置本地vanity服务地址
echo "VANITY_SERVICE_URL=http://localhost:8002" >> .env
```

3. **重启API服务**
```bash
# 如果使用docker-compose
docker-compose restart api

# 如果使用systemd
systemctl restart your-api-service
```

### 方案2：使用直接IP地址

如果vanity服务部署在其他服务器上：

```bash
# 设置为直接IP地址
export VANITY_SERVICE_URL=http://YOUR_VANITY_SERVER_IP:8002

# 或在.env文件中设置
VANITY_SERVICE_URL=http://YOUR_VANITY_SERVER_IP:8002
```

### 方案3：解决SSL证书验证问题

如果必须使用HTTPS连接但遇到SSL证书问题：

1. **临时禁用SSL验证（仅用于测试）**
```bash
# 在.env文件中添加
VERIFY_SSL=false
```

2. **安装证书**
```bash
# 更新证书库
apt-get update && apt-get install ca-certificates

# 如果使用自签名证书，添加到信任列表
cp your-cert.crt /usr/local/share/ca-certificates/
update-ca-certificates
```

3. **使用环境变量控制**
```bash
# 临时禁用Python的SSL验证
export PYTHONHTTPSVERIFY=0

# 或在代码中已经实现的环境变量
export VERIFY_SSL=false
```

### 方案4：解决网络问题

1. **检查DNS设置**
```bash
# 添加公共DNS
echo "nameserver 8.8.8.8" >> /etc/resolv.conf
echo "nameserver 1.1.1.1" >> /etc/resolv.conf
```

2. **检查防火墙设置**
```bash
# 检查iptables规则
iptables -L -n

# 如果使用ufw
ufw status

# 允许出站HTTPS连接
ufw allow out 443/tcp
```

## 诊断工具使用

### 1. 运行连接测试
```bash
cd server
python test_vanity_connection.py

# 测试自定义URL
python test_vanity_connection.py http://your-custom-url:8002
```

### 2. 运行自动修复工具
```bash
cd server
python fix_vanity_connection.py
```

## Docker Compose完整部署

如果还没有部署，使用以下步骤完整部署：

```bash
# 1. 克隆项目（如果还没有）
git clone https://github.com/your-repo/windows-clipboard-listener.git
cd windows-clipboard-listener

# 2. 配置环境变量
cd server
cp env.example .env
# 编辑.env文件，修改数据库密码等配置

# 3. 启动所有服务
cd ..
docker-compose up -d

# 4. 检查服务状态
docker-compose ps
```

## 环境变量配置示例

创建 `server/.env` 文件：

```env
# 数据库配置
DB_HOST=localhost
DB_PORT=3306
DB_USER=clipboard
DB_PASSWORD=your_secure_password
DB_NAME=clipboard

# JWT密钥
JWT_SECRET=your-secret-key-here

# Vanity服务配置（使用本地服务）
VANITY_SERVICE_URL=http://localhost:8002

# 服务端口
PORT=8001
```

## 验证修复

修复后，可以通过以下方式验证：

1. **测试API健康状态**
```bash
curl http://localhost:8001/health
```

2. **测试vanity服务**
```bash
curl http://localhost:8002/
```

3. **查看服务日志**
```bash
# Docker日志
docker-compose logs -f api
docker-compose logs -f vanity-service

# 系统日志
journalctl -u your-api-service -f
```

## 常见问题

### Q: 为什么本地可以连接但VPS不行？
A: 通常是因为：
- VPS网络限制（防火墙、安全组）
- DNS解析问题
- Cloudflare Tunnel的地域限制

### Q: 是否必须使用Docker？
A: 不是必须的，但Docker能确保环境一致性。也可以直接运行Python服务：
```bash
cd vanity-service
pip install -r requirements.txt
python main.py
```

### Q: 如何提高vanity服务性能？
A: 
- 使用GPU版本（需要NVIDIA GPU）
- 增加超时时间
- 部署多个实例负载均衡

## 联系支持

如果问题仍然存在，请提供：
1. `python test_vanity_connection.py` 的输出
2. 服务日志
3. VPS的网络配置信息
