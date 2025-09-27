# GPU工具自动下载脚本

Write-Host "`n=========================================" -ForegroundColor Cyan
Write-Host "GPU加速工具下载脚本" -ForegroundColor Cyan
Write-Host "=========================================`n" -ForegroundColor Cyan

# 创建目录
$toolsPath = "vanity-service\gpu_tools"
if (!(Test-Path $toolsPath)) {
    New-Item -Path $toolsPath -ItemType Directory -Force | Out-Null
}

cd $toolsPath

# 下载profanity2
Write-Host "[1/2] 下载 profanity2 (ETH/BNB GPU工具)..." -ForegroundColor Yellow
try {
    $profanityUrl = "https://github.com/1inch/profanity2/releases/download/v1.1.0/profanity2.exe"
    $profanityPath = "profanity2.exe"
    
    if (Test-Path $profanityPath) {
        Write-Host "profanity2.exe 已存在，跳过下载" -ForegroundColor Green
    } else {
        Write-Host "正在下载..." -ForegroundColor White
        Invoke-WebRequest -Uri $profanityUrl -OutFile $profanityPath -UseBasicParsing
        Write-Host "✓ profanity2.exe 下载完成!" -ForegroundColor Green
    }
} catch {
    Write-Host "✗ profanity2 下载失败: $_" -ForegroundColor Red
    Write-Host "请手动下载: https://github.com/1inch/profanity2/releases" -ForegroundColor Yellow
}

Write-Host ""

# 下载VanitySearch
Write-Host "[2/2] 下载 VanitySearch (BTC GPU工具)..." -ForegroundColor Yellow
try {
    # VanitySearch最新版本
    $vanityUrl = "https://github.com/JeanLucPons/VanitySearch/releases/download/1.19/VanitySearch_1.19.zip"
    $vanityZip = "VanitySearch.zip"
    $vanityPath = "VanitySearch.exe"
    
    if (Test-Path $vanityPath) {
        Write-Host "VanitySearch.exe 已存在，跳过下载" -ForegroundColor Green
    } else {
        Write-Host "正在下载..." -ForegroundColor White
        Invoke-WebRequest -Uri $vanityUrl -OutFile $vanityZip -UseBasicParsing
        
        # 解压
        Write-Host "正在解压..." -ForegroundColor White
        Expand-Archive -Path $vanityZip -DestinationPath . -Force
        
        # 移动exe到当前目录
        if (Test-Path "VanitySearch_1.19\VanitySearch.exe") {
            Move-Item "VanitySearch_1.19\VanitySearch.exe" . -Force
            Remove-Item "VanitySearch_1.19" -Recurse -Force
        }
        
        Remove-Item $vanityZip -Force
        Write-Host "✓ VanitySearch.exe 下载完成!" -ForegroundColor Green
    }
} catch {
    Write-Host "✗ VanitySearch 下载失败: $_" -ForegroundColor Red
    Write-Host "请手动下载: https://github.com/JeanLucPons/VanitySearch/releases" -ForegroundColor Yellow
}

Write-Host ""

# 检查文件
Write-Host "检查GPU工具..." -ForegroundColor Cyan
$tools = @("profanity2.exe", "VanitySearch.exe")
$allFound = $true

foreach ($tool in $tools) {
    if (Test-Path $tool) {
        Write-Host "✓ $tool" -ForegroundColor Green
    } else {
        Write-Host "✗ $tool 未找到" -ForegroundColor Red
        $allFound = $false
    }
}

if ($allFound) {
    Write-Host "`n✓ 所有GPU工具已准备就绪!" -ForegroundColor Green
    
    # 测试GPU工具
    Write-Host "`n测试GPU工具..." -ForegroundColor Cyan
    
    # 测试profanity2
    Write-Host "`n测试 profanity2:" -ForegroundColor Yellow
    try {
        $output = & ".\profanity2.exe" --help 2>&1 | Select-Object -First 5
        Write-Host $output -ForegroundColor Gray
        Write-Host "✓ profanity2 可以运行" -ForegroundColor Green
    } catch {
        Write-Host "✗ profanity2 无法运行" -ForegroundColor Red
    }
    
    # 测试VanitySearch
    Write-Host "`n测试 VanitySearch:" -ForegroundColor Yellow
    try {
        $output = & ".\VanitySearch.exe" -h 2>&1 | Select-Object -First 5
        Write-Host $output -ForegroundColor Gray
        Write-Host "✓ VanitySearch 可以运行" -ForegroundColor Green
    } catch {
        Write-Host "✗ VanitySearch 无法运行" -ForegroundColor Red
    }
} else {
    Write-Host "`n请手动下载缺失的工具" -ForegroundColor Yellow
}

cd ..\..\
Write-Host "`n完成！按任意键退出..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
