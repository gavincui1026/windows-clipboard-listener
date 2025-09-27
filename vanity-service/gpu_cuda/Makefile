# Makefile for TRON GPU Generator

# CUDA路径
CUDA_PATH ?= /usr/local/cuda

# 编译器
NVCC = $(CUDA_PATH)/bin/nvcc
CXX = g++

# 编译选项
NVCCFLAGS = -O3 -Xcompiler -fPIC -shared
CXXFLAGS = -O3 -fPIC -shared

# 自动检测GPU架构
GPU_ARCH = $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | sed 's/\.//g')
ifeq ($(GPU_ARCH),)
    GPU_ARCH = 70  # 默认Pascal
endif

NVCCFLAGS += -arch=sm_$(GPU_ARCH)

# 目标
TARGET = tron_gpu.so

# 源文件
SOURCES = tron_gpu_generator.cu

# 构建规则
all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

test: $(TARGET)
	@echo "Testing CUDA library..."
	@python3 -c "import ctypes; lib = ctypes.CDLL('./$(TARGET)', ctypes.RTLD_GLOBAL); print('✅ Library loaded successfully')"

install: $(TARGET)
	@echo "Library built successfully: $(TARGET)"
	@echo "GPU Architecture: sm_$(GPU_ARCH)"

.PHONY: all clean test install
