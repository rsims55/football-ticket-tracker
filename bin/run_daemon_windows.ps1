# bin/run_daemon_windows.ps1
# Run from anywhere: .\bin\run_daemon_windows.ps1

$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

if (-not (Test-Path ".venv_win\Scripts\python.exe")) {
    Write-Error ".venv_win not found. Run .\bin\setup_windows.ps1 first."
    exit 1
}

$env:PYTHONPATH = "$Root\src"
Write-Host "Starting CFB-Tix daemon..."
& .\.venv_win\Scripts\python.exe -m cfb_tix run --no-gui
