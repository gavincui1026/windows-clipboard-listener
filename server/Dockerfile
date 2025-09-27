FROM python:3.10-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    default-libmysqlclient-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 初始化数据库（在启动时运行）
RUN echo '#!/bin/bash\npython db.py\nuvicorn main:app --host 0.0.0.0 --port ${PORT:-8001}' > /app/start.sh && \
    chmod +x /app/start.sh

# 暴露端口
EXPOSE 8001

# 启动命令
CMD ["/app/start.sh"]
