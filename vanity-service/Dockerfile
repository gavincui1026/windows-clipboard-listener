# 基础镜像
FROM python:3.10-slim AS base

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libgmp-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8002

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]

# ===== GPU版本（可选）=====
# FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS gpu
# 
# RUN apt-get update && apt-get install -y \
#     python3-pip \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*
# 
# WORKDIR /app
# COPY requirements.txt .
# RUN pip3 install --no-cache-dir -r requirements.txt
# RUN pip3 install pycuda
# 
# COPY . .
# 
# # 复制GPU工具
# COPY gpu_tools /app/gpu_tools
# 
# EXPOSE 8001
# CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
