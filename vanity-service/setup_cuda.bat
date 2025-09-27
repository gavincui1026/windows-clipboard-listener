@echo off
echo.
echo ============================================
echo 设置C++ CUDA GPU生成器
echo ============================================
echo.

REM 检查CUDA
nvcc --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 未检测到CUDA编译器 (nvcc)
    echo.
    echo 请安装CUDA Toolkit:
    echo https://developer.nvidia.com/cuda-downloads
    echo.
    pause
    exit /b 1
)

echo ✅ 检测到CUDA编译器

REM 检查Visual Studio
where cl >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  未检测到Visual Studio编译器
    echo    可能需要运行VS开发者命令提示符
)

echo.
echo 开始编译CUDA代码...
cd gpu_cuda

REM 编译
call build.bat

if exist tron_gpu.dll (
    echo.
    echo ✅ 编译成功！
    echo.
    echo 现在可以使用GPU生成器了:
    echo - 重启vanity-service
    echo - API会自动使用C++ CUDA加速
    echo.
    echo 预期性能: 1亿+地址/秒 (RTX 5070 Ti)
) else (
    echo.
    echo ❌ 编译失败
    echo.
    echo 可能的原因:
    echo 1. CUDA版本不匹配
    echo 2. 显卡架构不支持
    echo 3. 缺少Visual Studio
)

cd ..
pause
