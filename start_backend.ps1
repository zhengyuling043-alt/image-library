$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "虚拟环境不存在，请先运行 .\setup_backend.ps1"
}
Set-Location -LiteralPath $ProjectRoot
& $Python web_app.py --port 8765
