# bin/run_daemon_windows.ps1
# Run from anywhere: .\bin\run_daemon_windows.ps1

$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

if (-not (Test-Path ".venv_win\Scripts\python.exe")) {
    Write-Error ".venv_win not found. Run .\bin\setup_windows.ps1 first."
    exit 1
}

$env:PYTHONPATH = "$Root\src"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

# Clear stale lock file if no daemon is actually running
$LockFile = "$env:LOCALAPPDATA\cfb-tix\Logs\daemon.lock"
if (Test-Path $LockFile) {
    $locked = $false
    try {
        $pid = [int](Get-Content $LockFile -ErrorAction Stop)
        $locked = (Get-Process -Id $pid -ErrorAction Stop) -ne $null
    } catch { }
    if (-not $locked) {
        Remove-Item $LockFile -Force -ErrorAction SilentlyContinue
    }
}

& .\.venv_win\Scripts\python.exe -m cfb_tix run --no-gui
