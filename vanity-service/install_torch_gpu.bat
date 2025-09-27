@echo off
echo.
echo ============================================
echo 安装PyTorch GPU版本
echo ============================================
echo.

echo [1] 卸载CPU版本的PyTorch...
pip uninstall -y torch torchvision torchaudio

echo.
echo [2] 安装GPU版本的PyTorch (CUDA 12.1)...
echo 这可能需要几分钟，文件较大（~2.5GB）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [3] 验证安装...
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')"

echo.
echo ============================================
echo 安装完成！
echo ============================================
pause
