# packaging/windows/install_win.ps1
# Idempotent installer for current user:
# - Creates %LocalAppData%\cfb-tix\venv and installs the app in editable mode from {AppDir}\app
# - Registers "CFB Tickets" Task Scheduler job at user logon to run headless daemon
# - Creates Start Menu shortcut "CFB Tickets (GUI)" pointing to venv Scripts\cfb-tix-gui(.exe)
param(
  [string]$AppDir = "$(Split-Path -Parent $MyInvocation.MyCommand.Path)",
  [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

$pkgName = "cfb-tix"
$appName = "CFB Ticket Price Tracker"
$taskName = "CFB Tickets"
$localBase = Join-Path $env:LocalAppData $pkgName
$venvDir = Join-Path $localBase "venv"
$appDst  = Join-Path $localBase "app"
$iconPath = Join-Path $AppDir "assets\icons\cfb-tix.ico"
$pyExe = Join-Path $venvDir "Scripts\python.exe"
$guiExe = Join-Path $venvDir "Scripts\cfb-tix-gui.exe"  # console_scripts stub
if (-not (Test-Path $guiExe)) {
  # when running from python -m, console_scripts may be .exe or no .exe depending on pip
  $guiExe = Join-Path $venvDir "Scripts\cfb-tix-gui"
}
$startMenuDir = Join-Path $env:ProgramData "Microsoft\Windows\Start Menu\Programs"
$shortcutPath = Join-Path $startMenuDir "CFB Tickets (GUI).lnk"

function Ensure-Shortcut {
  param([string]$Target, [string]$Shortcut, [string]$Icon)
  $shell = New-Object -ComObject WScript.Shell
  $sc = $shell.CreateShortcut($Shortcut)
  $sc.TargetPath = $Target
  $sc.WorkingDirectory = (Split-Path $Target -Parent)
  if (Test-Path $Icon) { $sc.IconLocation = $Icon }
  $sc.WindowStyle = 1
  $sc.Description = "Open the CFB Ticket Price Tracker GUI"
  $sc.Save()
}

if ($Uninstall) {
  try {
    schtasks.exe /Delete /TN "$taskName" /F | Out-Null
  } catch { }
  if (Test-Path $shortcutPath) { Remove-Item -Force "$shortcutPath" }
  Write-Host "✅ Unregistered scheduled task and removed Start Menu shortcut."
  exit 0
}

# Create base dirs
New-Item -ItemType Directory -Force -Path $localBase | Out-Null
New-Item -ItemType Directory -Force -Path $appDst     | Out-Null
New-Item -ItemType Directory -Force -Path $startMenuDir | Out-Null

# Copy/refresh app source into user-writable location (exclude junk)
$copyArgs = @("/E","/PURGE","/NFL","/NDL","/NJH","/NJS","/NC","/NS","/NP",
              "/XF","*.pyc",
              "/XD",".git","dist","build","__pycache__",".github","*.egg-info")
robocopy "$AppDir\app" "$appDst" $copyArgs | Out-Null

# Ensure Python & venv
$python = (Get-Command python -ErrorAction SilentlyContinue) ?? (Get-Command py -ErrorAction SilentlyContinue)
if (-not $python) { throw "Python is required in PATH." }

if (-not (Test-Path $venvDir)) {
  & $python.Path -m venv "$venvDir"
}
& "$pyExe" -m pip install --upgrade --disable-pip-version-check pip setuptools wheel

Push-Location $appDst
& "$pyExe" -m pip install -e .
Pop-Location

# Register (replace) scheduled task at user logon
try {
  schtasks.exe /Delete /TN "$taskName" /F | Out-Null
} catch { }

$action  = "powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -Command `"& '$pyExe' -m cfb_tix --no-gui`""
schtasks.exe /Create /TN "$taskName" /TR "$action" /SC ONLOGON /RL LIMITED /F | Out-Null

# Create Start Menu shortcut to GUI
Ensure-Shortcut -Target "$guiExe" -Shortcut "$shortcutPath" -Icon "$iconPath"

Write-Host "✅ $appName installed for user."
Write-Host "• Background task '$taskName' registered (runs headless on logon)."
Write-Host "• Start Menu → 'CFB Tickets (GUI)'."
