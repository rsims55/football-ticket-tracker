# packaging/windows/install_win.ps1
# Idempotent installer for current user:
# - Requires secrets.txt at the PROJECT ROOT (same folder as pyproject.toml)
# - If missing, prompts to create a template, opens in Notepad, and waits
# - Mirrors the project into %LocalAppData%\cfb-tix\app
# - Creates %LocalAppData%\cfb-tix\venv and installs the app (editable) into that venv
# - Registers "CFB Tickets" Task Scheduler job at user logon to run the headless daemon
# - Creates Start Menu shortcut "CFB Tickets (GUI)" pointing to venv Scripts\cfb-tix-gui(.exe)
# - Immediately runs daily_snapshot and weekly_update once after install
param(
  [string]$AppDir = "$(Split-Path -Parent $MyInvocation.MyCommand.Path)",
  [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

# ---- Names/paths ----
$pkgName   = "cfb-tix"
$appName   = "CFB Ticket Price Tracker"
$taskName  = "CFB Tickets"
$localBase = Join-Path $env:LocalAppData $pkgName
$venvDir   = Join-Path $localBase "venv"
$appDst    = Join-Path $localBase "app"

# Start Menu (per-user)
$startMenuDir = Join-Path $env:AppData "Microsoft\Windows\Start Menu\Programs"
$shortcutPath = Join-Path $startMenuDir "CFB Tickets (GUI).lnk"

# Icons (repo or packaged)
$iconPath = Join-Path (Join-Path $AppDir "..") "assets\icons\cfb-tix.ico"
if (-not (Test-Path $iconPath)) { $iconPath = Join-Path $appDst "assets\icons\cfb-tix.ico" }

# Python executables inside venv
$pyExe  = Join-Path $venvDir "Scripts\python.exe"
$pyWExe = Join-Path $venvDir "Scripts\pythonw.exe"  # preferred for no console

function Ensure-Shortcut {
  param([string]$Target, [string]$Shortcut, [string]$Icon)
  New-Item -ItemType Directory -Force -Path (Split-Path $Shortcut -Parent) | Out-Null
  $shell = New-Object -ComObject WScript.Shell
  $sc = $shell.CreateShortcut($Shortcut)
  $sc.TargetPath = $Target
  $sc.WorkingDirectory = (Split-Path $Target -Parent)
  if (Test-Path $Icon) { $sc.IconLocation = $Icon }
  $sc.WindowStyle = 1
  $sc.Description = "Open the CFB Ticket Price Tracker GUI"
  $sc.Save()
}

function Read-KeyValueFile {
  param([string]$Path)
  $map = @{}
  if (-not (Test-Path $Path)) { throw "Secrets file not found: $Path" }
  $lines = Get-Content -Raw $Path -ErrorAction Stop -Encoding UTF8
  foreach ($line in ($lines -split "`r?`n")) {
    $t = $line.Trim()
    if (-not $t -or $t.StartsWith("#")) { continue }
    $eq = $t.IndexOf("=")
    if ($eq -lt 1) { continue }
    $k = $t.Substring(0, $eq).Trim()
    $v = $t.Substring($eq+1).Trim()
    if ($v.StartsWith('"') -and $v.EndsWith('"')) { $v = $v.Substring(1, $v.Length-2) }
    if ($v.StartsWith("'") -and $v.EndsWith("'")) { $v = $v.Substring(1, $v.Length-2) }
    if ($k) { $map[$k] = $v }
  }
  return $map
}

# ---- Uninstall path ----
if ($Uninstall) {
  try { schtasks.exe /Delete /TN "$taskName" /F | Out-Null } catch { }
  if (Test-Path $shortcutPath) { Remove-Item -Force "$shortcutPath" }
  Write-Host "✅ Unregistered scheduled task and removed Start Menu shortcut."
  exit 0
}

# ---- Locate PROJECT ROOT (where pyproject.toml lives) ----
# Try: repo two levels up from packaging\windows\ ; else current working dir
$repoGuess = Resolve-Path (Join-Path $AppDir "..\..") -ErrorAction SilentlyContinue
$srcRoot = $null
if ($repoGuess -and (Test-Path (Join-Path $repoGuess "pyproject.toml"))) {
  $srcRoot = "$repoGuess"
} elseif (Test-Path (Join-Path (Get-Location).Path "pyproject.toml")) {
  $srcRoot = (Get-Location).Path
} else {
  throw "Could not locate project root (pyproject.toml). Run this script from the repo."
}

# ---- Require secrets.txt at PROJECT ROOT; prompt to create if missing ----
$secretsPath = Join-Path $srcRoot "secrets.txt"
if (-not (Test-Path $secretsPath)) {
  Write-Host "secrets.txt not found at: $secretsPath" -ForegroundColor Yellow
  $ans = Read-Host "Create a template secrets.txt there now? (Y/N)"
  if ($ans -match '^[Yy]') {
    $template = @"
# secrets.txt — configuration for CFB Ticket Price Tracker
# Place this file at the PROJECT ROOT (same folder as pyproject.toml).
# Lines are KEY=VALUE (compatible with python-dotenv). Leave blank to fill later.

# GitHub token (either key name works; SNAP_GH_TOKEN is normalized)
SNAP_GH_TOKEN=
# GITHUB_TOKEN=

# CollegeFootballData API key (used by schedule/weekly jobs)
CFD_API_KEY=

# Email settings (for notifications/exports if used)
GMAIL_APP_PASSWORD=
GMAIL_ADDRESS=
TO_EMAIL=

# Tips:
# - Keep this file OUT of version control (add 'secrets.txt' to .gitignore).
# - You can commit a 'secrets.example.txt' with empty values for teammates.
"@
    $template | Out-File -Encoding ASCII -Force $secretsPath
    Write-Host "Opening secrets.txt for editing..." -ForegroundColor Cyan
    notepad $secretsPath
    Write-Host "Save the file, then press Enter to continue..."
    [void][System.Console]::ReadLine()
  } else {
    throw "Please create secrets.txt at the project root and rerun. Expected at: $secretsPath"
  }
}

# ---- Parse secrets and set env ----
$secrets = Read-KeyValueFile -Path $secretsPath
if (-not $secrets.ContainsKey("SNAP_GH_TOKEN") -and $secrets.ContainsKey("GITHUB_TOKEN")) {
  $secrets["SNAP_GH_TOKEN"] = $secrets["GITHUB_TOKEN"]
}
if (-not $secrets.ContainsKey("GITHUB_TOKEN") -and $secrets.ContainsKey("SNAP_GH_TOKEN")) {
  $secrets["GITHUB_TOKEN"] = $secrets["SNAP_GH_TOKEN"]
}
$keys = @("SNAP_GH_TOKEN","GITHUB_TOKEN","CFD_API_KEY","GMAIL_APP_PASSWORD","GMAIL_ADDRESS","TO_EMAIL")
foreach ($k in $keys) {
  if ($secrets.ContainsKey($k) -and $secrets[$k]) {
    [Environment]::SetEnvironmentVariable($k, $secrets[$k], "User")
  }
}

# ---- Create base dirs ----
New-Item -ItemType Directory -Force -Path $localBase,$appDst,$startMenuDir | Out-Null

# ---- Mirror sources into user-writable appDst (exclude junk) ----
$rcArgs = @(
  $srcRoot, $appDst,
  '/MIR',
  '/XD', '.git', '.venv', 'venv', 'packaging\dist', 'packaging\.build_ext4_work', '__pycache__',
  '/XF', '*.pyc', '*.log'
)
robocopy @rcArgs | Out-Null

# ---- Write .env next to the installed app ----
$envPath = Join-Path $appDst ".env"
$envOut = foreach ($k in $keys) { if ($secrets.ContainsKey($k)) { "$k=$($secrets[$k])" } }
$envOut -join "`r`n" | Out-File -Encoding ASCII -Force $envPath
try { icacls "$envPath" /inheritance:r /grant:r "$env:USERNAME:(R,W)" | Out-Null } catch { }

# ---- Ensure Python & venv ----
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { $python = Get-Command py -ErrorAction SilentlyContinue }
if (-not $python) { throw "Python is required in PATH." }

if (-not (Test-Path $venvDir)) {
  & $python.Path -m venv "$venvDir"
}
& "$pyExe" -m pip install --upgrade --disable-pip-version-check pip setuptools wheel

# ---- Install app into that venv (editable) ----
Push-Location $appDst
& "$pyExe" -m pip install -e .
Pop-Location

# ---- Immediately run daily_snapshot and weekly_update once ----
$daily  = Join-Path $appDst "src\builders\daily_snapshot.py"
$weekly = Join-Path $appDst "src\builders\weekly_update.py"
if (-not (Test-Path $weekly)) { $weekly = Join-Path $appDst "src\reports\generate_weekly_report.py" }

Write-Host ">> Running daily_snapshot.py once..."
try { & "$pyExe" "$daily" } catch { Write-Host "WARN: daily_snapshot failed: $($_.Exception.Message)" }

Write-Host ">> Running weekly_update once..."
try { & "$pyExe" "$weekly" } catch { Write-Host "WARN: weekly_update failed: $($_.Exception.Message)" }

# ---- Register (replace) scheduled task at user logon ----
try { schtasks.exe /Delete /TN "$taskName" /F | Out-Null } catch { }
$runner = $pyWExe
if (-not (Test-Path $pyWExe)) { $runner = $pyExe }
$action = "`"$runner`" -m cfb_tix --no-gui"
schtasks.exe /Create /TN "$taskName" /TR $action /SC ONLOGON /RL LIMITED /F | Out-Null

# ---- Create Start Menu shortcut to GUI ----
$guiExe = Join-Path $venvDir "Scripts\cfb-tix-gui.exe"
if (-not (Test-Path $guiExe)) { $guiExe = Join-Path $venvDir "Scripts\cfb-tix-gui" }
Ensure-Shortcut -Target "$guiExe" -Shortcut "$shortcutPath" -Icon "$iconPath"

Write-Host "✅ $appName installed for user."
Write-Host "• Background task '$taskName' registered (runs headless on logon)."
Write-Host "• Start Menu → 'CFB Tickets (GUI)'."
Write-Host "• Using secrets file at: $secretsPath"
