$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PipCache = Join-Path $ProjectRoot ".runtime\pip-cache"

if (-not (Test-Path -LiteralPath $Python)) {
    py -3.12 -m venv (Join-Path $ProjectRoot ".venv")
}
New-Item -ItemType Directory -Force -Path $PipCache | Out-Null
$env:PIP_CACHE_DIR = $PipCache
& $Python -m pip install --upgrade pip
& $Python -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
& $Python -m unittest discover -v
