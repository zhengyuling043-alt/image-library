param(
    [string]$Base = "origin/main",
    [string]$Branch = "backend/b-model-index",
    [string]$AuthorEmail = "2026147135@qq.com"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $repo
$range = "${Base}..${Branch}"

Write-Host "=== Backend Contribution Evidence ===" -ForegroundColor Cyan
Write-Host "Repository: $repo"
Write-Host "Baseline: $Base"
Write-Host "Branch: $Branch"
Write-Host "Author email: $AuthorEmail"

git rev-parse --verify $Base | Out-Null
git rev-parse --verify $Branch | Out-Null

Write-Host "`n[1] Personal commits" -ForegroundColor Yellow
git log $range --author=$AuthorEmail --date=short `
    --pretty=format:"%h  %ad  %an <%ae>  %s"

Write-Host "`n`n[2] Changed-file statistics" -ForegroundColor Yellow
git diff --stat $range

Write-Host "`n[3] Files added from scratch" -ForegroundColor Yellow
git diff --name-only --diff-filter=A $range

Write-Host "`n[4] Backend source line statistics" -ForegroundColor Yellow
git diff --numstat $range -- "*.py" "*.ps1"

Write-Host "`n[5] Working-tree status" -ForegroundColor Yellow
git status --short --branch

