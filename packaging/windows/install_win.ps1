<# Builds a self-contained Windows zip with:
   - /app                 → repo (clean copy, excludes junk)
   - /assets/...          → windows icon (if present)
   - /install_win.ps1     → idempotent app installer (venv + autostart + Start Menu)
   - /install_sync_win.ps1→ schedules daily CSV sync + first-time pull (prompts for GH token)

   Output: packaging\dist\cfb-tix-win.zip
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# --- repo roots ---
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$REPO_ROOT  = Resolve-Path (Join-Path $SCRIPT_DIR "..\..")
$DIST_DIR   = Join-Path $REPO_ROOT "packaging\dist"
$WORK_DIR   = Join-Path $REPO_ROOT ".build_win_work"
$STAGE      = Join-Path $WORK_DIR "root"
$ZIP_PATH   = Join-Path $DIST_DIR "cfb-tix-win.zip"

# --- clean work ---
if (Test-Path $WORK_DIR) { Remove-Item -Recurse -Force $WORK_DIR }
New-Item -ItemType Directory -Force -Path $STAGE | Out-Null
New-Item -ItemType Directory -Force -Path $DIST_DIR | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $STAGE "app") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $STAGE "assets\icons") | Out-Null

# --- copy repo into /app (exclude junk) ---
$src = $REPO_ROOT
$dst = Join-Path $STAGE "app"
# robocopy exit codes: 0,1 = success; 2+ also success w/ extra, so don't Stop on non-zero.
$copyArgs = @(
  "`"$src`"", "`"$dst`"", "/E", "/PURGE", "/NFL", "/NDL", "/NJH", "/NJS", "/NC", "/NS", "/NP",
  "/XF","*.pyc",
  "/XD",".git",".github","packaging\dist","dist","build","__pycache__","*.egg-info",".venv",".mypy_cache",".pytest_cache"
)
$null = Start-Process -FilePath robocopy.exe -ArgumentList $copyArgs -Wait -NoNewWindow -PassThru

# --- copy Windows icon if present ---
$ico = Join-Path $REPO_ROOT "assets\icons\cfb-tix.ico"
if (Test-Path $ico) {
  Copy-Item $ico (Join-Path $STAGE "assets\icons\cfb-tix.ico")
}

# --- drop installers into stage root ---
# 1) Use your existing install_win.ps1 from the repo to keep single source of truth
$installerSrc = Join-Path $REPO_ROOT "packaging\windows\install_win.ps1"
if (-not (Test-Path $installerSrc)) {
  throw "Expected installer at packaging\windows\install_win.ps1 not found."
}
Copy-Item $installerSrc (Join-Path $STAGE "install_win.ps1")

# 2) Generate install_sync_win.ps1 (Windows analog of install_sync.sh)
$syncScript = @'
# install_sync_win.ps1 — per-user daily CSV sync + first-time pull
param(
  [string]$RepoDir   = "$env:LOCALAPPDATA\cfb-tix\app",
  [string]$PythonBin = "$env:LOCALAPPDATA\cfb-tix\venv\Scripts\python.exe",
  [string]$RunTime   = "06:10"  # local time, HH:mm
)

$ErrorActionPreference = "Stop"
$taskName = "cfb-tix-sync"
$envFile  = Join-Path $env:APPDATA "cfb-tix\env"
$envDir   = Split-Path $envFile -Parent

# ---- Prompt for GH token (once) and store securely ----
New-Item -ItemType Directory -Force -Path $envDir | Out-Null
if (-not (Test-Path $envFile) -or -not (Get-Content $envFile | Select-String -SimpleMatch "GH_TOKEN=")) {
  Write-Host "🔑 No GitHub token found."
  $token = Read-Host -AsSecureString "Paste your GitHub access token (leave blank to skip uploads)"
  $plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
             [Runtime.InteropServices.Marshal]::SecureStringToBSTR($token))
  if ($plain) {
    "GH_TOKEN=$plain" | Out-File -Encoding utf8 -FilePath $envFile -Force
    Write-Host "✅ Saved token to $envFile"
  } else {
    "GH_TOKEN=" | Out-File -Encoding utf8 -FilePath $envFile -Force
    Write-Host "⚠️ Skipping token setup. Uploads will be disabled."
  }
}

# ---- Create a small wrapper that loads env then runs the sync step ----
$binDir = Join-Path $env:LOCALAPPDATA "cfb-tix\bin"
New-Item -ItemType Directory -Force -Path $binDir | Out-Null
$runner = Join-Path $binDir "run_sync.ps1"
@"
`$ErrorActionPreference = 'Stop'
`$envVars = Get-Content -ErrorAction SilentlyContinue '$envFile' | Where-Object { $_ -match '=' }
foreach (`$kv in `$envVars) {
  `$name,`$val = `$kv -split '=',2
  if (`$name) { [Environment]::SetEnvironmentVariable(`$name, `$val, 'Process') }
}
& '$PythonBin' '$RepoDir\scripts\sync_snapshots.py' pull_push
"@ | Out-File -Encoding utf8 -FilePath $runner -Force

# ---- (Re)create daily scheduled task at $RunTime ----
try { schtasks /Delete /TN "$taskName" /F | Out-Null } catch { }
schtasks /Create /TN "$taskName" /TR "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$runner`"" `
  /SC DAILY /ST $RunTime /RL LIMITED /F | Out-Null

# ---- First-time pull (download only; non-fatal) ----
try { & $PythonBin "$RepoDir\scripts\sync_snapshots.py" pull } catch { }

Write-Host "✅ Installed user task $taskName at $RunTime daily."
Write-Host "   Repo: $RepoDir"
Write-Host "   Token file: $envFile"
'@
Set-Content -Path (Join-Path $STAGE "install_sync_win.ps1") -Value $syncScript -Encoding UTF8
# --- zip the staged payload ---
if (Test-Path $ZIP_PATH) { Remove-Item -Force $ZIP_PATH }
Compress-Archive -Path (Join-Path $STAGE '*') -DestinationPath $ZIP_PATH

Write-Host "🎉 Built $ZIP_PATH"
