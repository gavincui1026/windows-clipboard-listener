@echo off
echo.
echo ===========================================
echo 下载GPU加速工具
echo ===========================================
echo.

cd vanity-service\gpu_tools

echo [1/2] 下载 profanity2 (ETH/BNB GPU工具)...
echo 请手动下载:
echo https://github.com/1inch/profanity2/releases/latest
echo 下载 profanity2.exe 到 vanity-service\gpu_tools\
echo.

echo [2/2] 下载 VanitySearch (BTC GPU工具)...
echo 请手动下载:
echo https://github.com/JeanLucPons/VanitySearch/releases
echo 下载 VanitySearch.exe 到 vanity-service\gpu_tools\
echo.

echo 下载完成后，文件结构应该是:
echo vanity-service\
echo   gpu_tools\
echo     profanity2.exe
echo     VanitySearch.exe
echo.

echo 提示：也可以使用PowerShell自动下载:
echo.
echo PowerShell下载命令:
echo # Profanity2
echo Invoke-WebRequest -Uri "https://github.com/1inch/profanity2/releases/download/v1.0.2/profanity2.exe" -OutFile "profanity2.exe"
echo.
echo # VanitySearch (需要手动从releases页面获取最新链接)
echo.

cd ..\..
pause
