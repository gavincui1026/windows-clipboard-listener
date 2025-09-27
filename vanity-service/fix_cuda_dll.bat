@echo off
echo.
echo ============================================
echo 修复CUDA DLL加载问题
echo ============================================
echo.

REM 检查CUDA安装
echo [1] 检查CUDA安装...
nvcc --version 2>nul
if %errorlevel% neq 0 (
    echo.
    echo ✗ 未检测到CUDA Toolkit
    echo.
    echo 请先安装CUDA Toolkit:
    echo https://developer.nvidia.com/cuda-downloads
    echo.
    echo 推荐版本: CUDA 12.1 或更高
    pause
    exit /b 1
)

echo.
echo [2] 检查CUDA路径...
echo CUDA_PATH=%CUDA_PATH%

REM 添加CUDA到PATH
echo.
echo [3] 添加CUDA库到系统PATH...
set "CUDA_BIN=%CUDA_PATH%\bin"
set "CUDA_LIB=%CUDA_PATH%\lib\x64"

echo 添加以下路径到PATH:
echo - %CUDA_BIN%
echo - %CUDA_LIB%

REM 检查curand.dll是否存在
echo.
echo [4] 检查CUDA库文件...
if exist "%CUDA_BIN%\curand64_*.dll" (
    echo ✓ 找到curand库
    dir /b "%CUDA_BIN%\curand64_*.dll"
) else (
    echo ✗ 未找到curand库
)

echo.
echo [5] 重新安装CuPy...
echo.
echo 卸载现有CuPy...
pip uninstall -y cupy-cuda11x cupy-cuda12x cupy

echo.
echo 检测CUDA版本并安装对应CuPy...
nvcc --version | findstr "release 12" >nul
if %errorlevel%==0 (
    echo 安装CuPy for CUDA 12.x...
    pip install cupy-cuda12x
) else (
    nvcc --version | findstr "release 11" >nul
    if %errorlevel%==0 (
        echo 安装CuPy for CUDA 11.x...
        pip install cupy-cuda11x
    ) else (
        echo 安装CuPy for CUDA 12.x (默认)...
        pip install cupy-cuda12x
    )
)

echo.
echo [6] 测试CuPy...
python -c "import cupy; print('CuPy导入成功！'); print(f'CUDA设备: {cupy.cuda.runtime.getDevice()}')" 2>nul
if %errorlevel%==0 (
    echo ✓ CuPy测试成功！
) else (
    echo ✗ CuPy测试失败
    echo.
    echo 可能的解决方案:
    echo 1. 重启电脑使环境变量生效
    echo 2. 手动添加CUDA路径到系统PATH
    echo 3. 安装Visual C++ Redistributables
    echo    https://aka.ms/vs/17/release/vc_redist.x64.exe
)

echo.
echo ============================================
echo 修复完成！
echo.
echo 如果问题仍然存在，请尝试：
echo 1. 重启电脑
echo 2. 运行 setup_gpu_fallback.bat 使用替代方案
echo ============================================
echo.
pause
