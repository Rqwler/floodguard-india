param(
    [int]$Port = 8000,
    [switch]$Open
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment in .venv ..." -ForegroundColor Cyan
    python -m venv .venv
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r requirements.txt
}

$env:PYTHONUTF8 = "1"
$Url = "http://127.0.0.1:$Port"

if ($Open) {
    Start-Job -ScriptBlock {
        param($Target)
        Start-Sleep -Seconds 4
        Start-Process $Target
    } -ArgumentList $Url | Out-Null
}

Write-Host ""
Write-Host "  FloodGuard India" -ForegroundColor Cyan
Write-Host "  Dashboard: $Url" -ForegroundColor White
Write-Host "  API docs : $Url/docs" -ForegroundColor DarkGray
Write-Host ""

& $VenvPython -m uvicorn app.main:app --host 127.0.0.1 --port $Port
