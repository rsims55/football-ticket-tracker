# scripts/start_daemon.ps1
# Launch the CFB-Tix daemon (push-only sync enabled), headless.

$ErrorActionPreference = "Stop"

# 1) repo root = parent of /scripts
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

# 2) env for daemon: enable push-only sync; commit only data,models; NEVER download release assets
$env:CFB_TIX_ENABLE_SYNC     = "1"
$env:CFB_TIX_COMMIT_SCOPE    = "data,models"
$env:CFB_TIX_USE_RELEASE_SYNC = "0"
$env:REPO_DATA_LOCK          = "1"
$env:REPO_ALLOW_NON_REPO_OUT = "0"

# 3) ensure src is on PYTHONPATH (so imports work anywhere)
$src = Join-Path $ROOT "src"
if ($env:PYTHONPATH) {
  if ($env:PYTHONPATH -notlike "*$src*") { $env:PYTHONPATH = "$src;$env:PYTHONPATH" }
} else {
  $env:PYTHONPATH = $src
}

# 4) choose a python (prefer local venv if present)
$PyCandidates = @(
  (Join-Path $ROOT ".venv\Scripts\python.exe"),
  (Join-Path $ROOT "venv\Scripts\python.exe"),
  "python",
  "py"
) | Where-Object { ($_ -in @("python","py")) -or (Test-Path $_) }
$Py = $PyCandidates[0]

# 5) launch hidden, let Python write its own rotating logs in %LOCALAPPDATA%\cfb-tix\Logs
$LOGDIR   = Join-Path $env:LOCALAPPDATA "cfb-tix\Logs"
New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null
$LAUNCHLOG = Join-Path $LOGDIR "daemon-launch.log"
"$(Get-Date -Format s) :: launching daemon with $Py" | Out-File -Append -FilePath $LAUNCHLOG

Start-Process -FilePath $Py `
  -ArgumentList "-m","cfb_tix.daemon","run","--no-gui" `
  -WorkingDirectory $ROOT `
  -WindowStyle Hidden
