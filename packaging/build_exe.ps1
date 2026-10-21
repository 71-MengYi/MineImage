# MineImage 一键打包脚本
# 用法：在仓库根目录执行 powershell -ExecutionPolicy Bypass -File packaging/build_exe.ps1

param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "== MineImage 一键打包 ==" -ForegroundColor Cyan
Write-Host "项目目录: $Root"

if (-not $SkipInstall) {
    python -c "import PyInstaller" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "未检测到 PyInstaller，正在安装..." -ForegroundColor Yellow
        python -m pip install --upgrade pyinstaller pillow
        if ($LASTEXITCODE -ne 0) { throw "依赖安装失败" }
    }
}

Write-Host "生成图标..." -ForegroundColor Cyan
python packaging/make_icon.py
if ($LASTEXITCODE -ne 0) { throw "图标生成失败" }

Write-Host "运行 PyInstaller..." -ForegroundColor Cyan
python -m PyInstaller --noconfirm --clean packaging/MineImage.spec
if ($LASTEXITCODE -ne 0) { throw "打包失败" }

$Dist = Join-Path $Root "dist"
New-Item -ItemType Directory -Force -Path (Join-Path $Dist "docs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Dist "operators") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Dist "sample_images") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Dist "examples") | Out-Null
Copy-Item -Recurse -Force (Join-Path $Root "docs\*") (Join-Path $Dist "docs")
Copy-Item -Recurse -Force (Join-Path $Root "examples\*") (Join-Path $Dist "examples")
Copy-Item -Force (Join-Path $Root "operators\README.md") (Join-Path $Dist "operators")
Copy-Item -Force (Join-Path $Root "operators\_template.py") (Join-Path $Dist "operators")
if (Test-Path (Join-Path $Root "sample_images")) {
    Copy-Item -Force (Join-Path $Root "sample_images\*") (Join-Path $Dist "sample_images")
}
Copy-Item -Force (Join-Path $Root "README.md") (Join-Path $Dist "README.md")
if (Test-Path (Join-Path $Root "LICENSE")) {
    Copy-Item -Force (Join-Path $Root "LICENSE") (Join-Path $Dist "LICENSE")
}

$Exe = Join-Path $Dist "MineImage.exe"
if (-not (Test-Path $Exe)) { throw "未找到产物：$Exe" }
$Size = [math]::Round((Get-Item $Exe).Length / 1MB, 1)
Write-Host "打包完成: $Exe ($Size MiB)" -ForegroundColor Green
Write-Host "用户算子目录: $Dist\operators" -ForegroundColor Green
