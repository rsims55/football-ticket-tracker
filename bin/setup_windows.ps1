# bin/setup_windows.ps1
# Run from the repo root: .\bin\setup_windows.ps1

$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

if (-not (Test-Path "requirements.txt")) {
    Write-Error "requirements.txt not found. Run from repo root."
    exit 1
}

Write-Host "Creating .venv_win..."
python -m venv .venv_win

Write-Host "Installing dependencies..."
& .\.venv_win\Scripts\pip.exe install --upgrade pip -q
& .\.venv_win\Scripts\pip.exe install -r requirements.txt

Write-Host ""
Write-Host "Setup complete. To start the daemon run: .\bin\run_daemon_windows.ps1"
