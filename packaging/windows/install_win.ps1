# packaging/windows/install_win.ps1
# Idempotent installer for current user:
# - Requires secrets.txt at the REPO ROOT (NOT next to pyproject.toml). Prompts to create if missing.
# - Mirrors repo's app/ folder into %LocalAppData%\cfb-tix\app
# - Creates %LocalAppData%\cfb-tix\venv and installs the app (editable) into that venv
# - Registers "CFB Tickets" Task Scheduler job at user logon to run the headless daemon
#   (fixed: uses `pythonw.exe -m cfb_tix run --no-gui`)
# - Creates Start Menu shortcut "CFB Tickets (GUI)" that calls a wrapper for reliable GUI launch
# - Immediately runs daily_snapshot and weekly_update once (weekly skipped if CFD_API_KEY is blank)

param(
  [switch]$Uninstall,
  [switch]$NoJobs   # pass -NoJobs to skip the first-run jobs
)

$ErrorActionPreference = "Stop"

# ---- Names/paths ----
$pkgName   = "cfb-tix"
$appName   = "CFB Ticket Price Tracker"
$taskName  = "CFB Tickets"

# Resolve repo root (script is at repo\packaging\windows\install_win.ps1)
try {
  $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
} catch {
  $RepoRoot = (Get-Location).Path
}

# IMPORTANT: your pyproject.toml is under app/, so we mirror from there
$RepoAppSrc = Join-Path $RepoRoot 'app'
if (-not (Test-Path (Join-Path $RepoAppSrc 'pyproject.toml'))) {
  throw "Expected pyproject.toml at: $RepoAppSrc. Run this from your cloned repo root where 'app\pyproject.toml' exists."
}

# Secrets live at the REPO ROOT (not next to pyproject)
$SecretsPath = Join-Path $RepoRoot 'secrets.txt'

$Base    = Join-Path $env:LocalAppData $pkgName
$VenvDir = Join-Path $Base 'venv'
$AppDst  = Join-Path $Base 'app'
$LogsDir = Join-Path $Base 'logs'
$BinDir  = Join-Path $Base 'bin'

$PyExe   = Join-Path $VenvDir 'Scripts\python.exe'
$PywExe  = Join-Path $VenvDir 'Scripts\pythonw.exe'

# GUI
$GuiPy       = Join-Path $AppDst 'src\gui\ticket_predictor_gui.py'
$IconPath    = Join-Path $AppDst 'assets\icons\cfb-tix.ico'
$GuiWrapper  = Join-Path $BinDir 'run_gui.cmd'
$UserStart   = Join-Path $env:AppData 'Microsoft\Windows\Start Menu\Programs'
$ShortcutU   = Join-Path $UserStart 'CFB Tickets (GUI).lnk'

# ---------- helpers ----------
function Ensure-Dir([string[]]$paths) {
  foreach ($p in $paths) { if (-not (Test-Path $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null } }
}

function Ensure-Shortcut([string]$Target,[string]$Args,[string]$Shortcut,[string]$Icon,[string]$WorkDir,[int]$WindowStyle=1) {
  New-Item -ItemType Directory -Force -Path (Split-Path $Shortcut -Parent) | Out-Null
  $shell = New-Object -ComObject WScript.Shell
  $sc = $shell.CreateShortcut($Shortcut)
  $sc.TargetPath = $Target
  $sc.Arguments  = $Args
  $sc.WorkingDirectory = $WorkDir
  if (Test-Path $Icon) { $sc.IconLocation = $Icon }
  $sc.WindowStyle = $WindowStyle
  $sc.Description = 'Open the CFB Ticket Price Tracker GUI'
  $sc.Save()
}

function Read-KeyValueFile([string]$Path) {
  $map = @{}
  if (-not (Test-Path $Path)) { return $map }
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

# ---------- uninstall ----------
if ($Uninstall) {
  try { schtasks /End /TN "$taskName" 2>$null | Out-Null } catch {}
  try { schtasks /Delete /TN "$taskName" /F 2>$null | Out-Null } catch {}
  if (Test-Path $ShortcutU) { Remove-Item $ShortcutU -Force }
  if (Test-Path $Base) { Remove-Item $Base -Recurse -Force -ErrorAction SilentlyContinue }
  Write-Host "✅ Uninstalled: scheduled task, user shortcut, and %LocalAppData%\$pkgName"
  exit 0
}

# ---------- ensure folders ----------
Ensure-Dir @($Base,$VenvDir,$AppDst,$LogsDir,$BinDir,$UserStart)

# ---------- require/collect secrets ----------
if (-not (Test-Path $SecretsPath)) {
@"
# secrets.txt — configuration for CFB Ticket Price Tracker
# Place this at the REPO ROOT (same folder that has the 'app\' directory).
# KEY=VALUE (compatible with python-dotenv). Leave blank to fill later.

# GitHub token (either key name works; SNAP_GH_TOKEN is normalized)
SNAP_GH_TOKEN=
# GITHUB_TOKEN=

# CollegeFootballData API key (used by weekly/schedule jobs)
CFD_API_KEY=

# Email settings (for notifications/exports if used)
GMAIL_APP_PASSWORD=
GMAIL_ADDRESS=
TO_EMAIL=
"@ | Out-File -Encoding ASCII -Force $SecretsPath
  Write-Host "⚠️  Created $SecretsPath. Please fill it now, save, then return here."
  Start-Process notepad $SecretsPath
  Read-Host "Press ENTER after you saved secrets.txt"
}
$secrets = Read-KeyValueFile $SecretsPath

# normalize alias both ways
if (-not $secrets.ContainsKey('SNAP_GH_TOKEN') -and $secrets.ContainsKey('GITHUB_TOKEN')) { $secrets['SNAP_GH_TOKEN'] = $secrets['GITHUB_TOKEN'] }
if (-not $secrets.ContainsKey('GITHUB_TOKEN') -and $secrets.ContainsKey('SNAP_GH_TOKEN')) { $secrets['GITHUB_TOKEN'] = $secrets['SNAP_GH_TOKEN'] }

# Persist as user env (optional but convenient)
$keys = @('SNAP_GH_TOKEN','GITHUB_TOKEN','CFD_API_KEY','GMAIL_APP_PASSWORD','GMAIL_ADDRESS','TO_EMAIL')
foreach ($k in $keys) { if ($secrets.ContainsKey($k) -and $secrets[$k]) { [Environment]::SetEnvironmentVariable($k,$secrets[$k],'User') } }

# ---------- mirror repo app -> installed app ----------
$robocopyArgs = @(
  '/MIR','/NFL','/NDL','/NJH','/NJS','/NC','/NS','/NP',
  '/XD','.git','__pycache__','.github','dist','build','*.egg-info','.venv','venv','packaging\dist','packaging\.build*',
  '/XF','*.pyc','*.log'
)
& robocopy "$RepoAppSrc" "$AppDst" $robocopyArgs | Out-Null

# ---------- write .env next to installed app ----------
$envPath = Join-Path $AppDst '.env'
$envOut = foreach ($k in $keys) { if ($secrets.ContainsKey($k)) { "$k=$($secrets[$k])" } }
$envOut -join "`r`n" | Out-File -Encoding ASCII -Force $envPath
try { icacls "$envPath" /inheritance:r /grant:r "$env:USERNAME:(R,W)" | Out-Null } catch {}

# ---------- choose Python (prefer 3.12, then 3.11, else 'python') ----------
$chosenPy = $null
$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pyLauncher) {
  $list = & py -0p
  if ($list -match '3\.12') { $chosenPy = 'py -3.12' }
  elseif ($list -match '3\.11') { $chosenPy = 'py -3.11' }
}
if (-not $chosenPy) {
  $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
  if ($pythonCmd) { $chosenPy = 'python' } else { throw "Python not found. Install Python 3.12 or 3.11." }
}

# ---------- venv + install editable ----------
if (-not (Test-Path $VenvDir)) {
  Write-Host "Creating venv with: $chosenPy"
  cmd /c "$chosenPy -m venv `"$VenvDir`""
}
& "$PyExe" -m pip install --upgrade --disable-pip-version-check pip setuptools wheel
Push-Location $AppDst
& "$PyExe" -m pip install -e .
Pop-Location

# ---------- first-run jobs ----------
if (-not $NoJobs) {
  $haveCFD = ($secrets.ContainsKey('CFD_API_KEY') -and $secrets['CFD_API_KEY'])
  try {
    Write-Host "▶ Running daily_snapshot once..."
    & "$PyExe" (Join-Path $AppDst 'src\builders\daily_snapshot.py')
  } catch { Write-Host "WARN: daily_snapshot failed: $($_.Exception.Message)" }

  try {
    if ($haveCFD) {
      Write-Host "▶ Running weekly_update once..."
      $weekly = Join-Path $AppDst 'src\builders\weekly_update.py'
      if (-not (Test-Path $weekly)) { $weekly = Join-Path $AppDst 'src\reports\generate_weekly_report.py' }
      & "$PyExe" $weekly
    } else {
      Write-Host "⏭️  Skipping weekly_update (CFD_API_KEY is blank)."
    }
  } catch { Write-Host "WARN: weekly_update failed: $($_.Exception.Message)" }
}

# ---------- register/replace scheduled task (FIXED command) ----------
try { schtasks /Delete /TN "$taskName" /F 2>$null | Out-Null } catch {}
$runner = (Test-Path $PywExe) ? $PywExe : $PyExe
# no ternary in older PS; do it explicitly:
if (-not (Test-Path $PywExe)) { $runner = $PyExe } else { $runner = $PywExe }
$action = "`"$runner`" -m cfb_tix run --no-gui"
schtasks /Create /TN "$taskName" /TR $action /SC ONLOGON /RL LIMITED /F | Out-Null

# ---------- GUI wrapper (reliable) + user shortcut ----------
@"
@echo off
setlocal
set "BASE=%LOCALAPPDATA%\cfb-tix"
set "APP=%BASE%\app"
set "PYW=%BASE%\venv\Scripts\pythonw.exe"
set "GUI=%APP%\src\gui\ticket_predictor_gui.py"
cd /d "%APP%"
start "" "%PYW%" "%GUI%"
"@ | Out-File -Encoding ASCII -Force $GuiWrapper

Ensure-Shortcut -Target $GuiWrapper -Args '' -Shortcut $ShortcutU -Icon $IconPath -WorkDir (Split-Path $GuiPy -Parent) -WindowStyle 1

# ---------- kick daemon once & show status ----------
schtasks /Run /TN "$taskName" 2>$null | Out-Null
Start-Sleep -Seconds 3

$log1 = Join-Path $Base 'logs\cfb_tix.log'
$log2 = Join-Path $env:LOCALAPPDATA 'cfb-tix\cfb-tix\Logs\cfb_tix.log'
$log = $(if (Test-Path $log1) { $log1 } elseif (Test-Path $log2) { $log2 } else { $null })

Write-Host ""
Write-Host "✅ $appName installed for user."
schtasks /Query /TN "$taskName"
Write-Host "Start Menu (user): $ShortcutU"
Write-Host "Log: " ($log ?? '<not found yet>')
if ($log) { Get-Content $log -Tail 20 }
