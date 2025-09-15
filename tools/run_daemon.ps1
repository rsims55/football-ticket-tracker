#Requires -Version 5.1
<#
One-script controller for the CFB-Tix Windows daemon.
Install/update/start/stop/uninstall, run headless once, view status/logs,
and kill/clean lock — all from this file.

USAGE (from repo root):
  powershell -ExecutionPolicy Bypass -File .\tools\run_daemon.ps1          # install+start (default)
  powershell -ExecutionPolicy Bypass -File .\tools\run_daemon.ps1 install
  powershell -ExecutionPolicy Bypass -File .\tools\run_daemon.ps1 start
  powershell -ExecutionPolicy Bypass -File .\tools\run_daemon.ps1 stop
  powershell -ExecutionPolicy Bypass -File .\tools\run_daemon.ps1 status
  powershell -ExecutionPolicy Bypass -File .\tools\run_daemon.ps1 log -Lines 200
  powershell -ExecutionPolicy Bypass -File .\tools\run_daemon.ps1 runonce   # headless one-off
  powershell -ExecutionPolicy Bypass -File .\tools\run_daemon.ps1 kill      # kill PID from lock
  powershell -ExecutionPolicy Bypass -File .\tools\run_daemon.ps1 cleanlock # delete stale lock
  powershell -ExecutionPolicy Bypass -File .\tools\run_daemon.ps1 uninstall
#>

param(
  [Parameter(Position=0)]
  [ValidateSet('install','start','stop','status','log','runonce','uninstall','_launch','kill','cleanlock')]
  [string]$Command = 'install',

  [int]$Lines = 200,

  # For manual runonce, defaults to headless unless explicitly overridden
  [switch]$NoGui
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Import-Module ScheduledTasks -ErrorAction SilentlyContinue | Out-Null

# ---------- Constants & Paths ----------
$TaskName    = 'CFB-Tix Daemon'
$AppName     = 'cfb-tix'
$ToolsDir    = Split-Path -Parent $PSCommandPath
$RepoRoot    = Split-Path -Parent $ToolsDir
$SrcDir      = Join-Path $RepoRoot 'src'
$VenvPythonw = Join-Path $RepoRoot '.venv\Scripts\pythonw.exe'
$UserLogDir  = Join-Path $env:LOCALAPPDATA "$AppName\Logs"
$MainLog     = Join-Path $UserLogDir 'cfb_tix.log'
$LockFile    = Join-Path $UserLogDir 'daemon.lock'

# Launcher (used by schtasks/startup/runkey fallbacks)
$LauncherDir  = Join-Path $env:LOCALAPPDATA "$AppName"
$LauncherCmd  = Join-Path $LauncherDir 'launch_daemon.cmd'

# Per-user Startup & Run key fallbacks
$StartupDir  = [Environment]::GetFolderPath('Startup')
$StartupLnk  = Join-Path $StartupDir 'CFB-Tix Daemon.lnk'
$RunKeyPath  = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run'
$RunKeyName  = 'CFB-Tix Daemon'

New-Item -Force -ItemType Directory -Path $UserLogDir | Out-Null

function Write-Note($msg) { Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg" }

# Default manual-run to headless unless explicitly overridden
if (($Command -eq 'runonce') -and (-not $PSBoundParameters.ContainsKey('NoGui'))) { $NoGui = $true }

function Get-Pythonw {
  if (Test-Path $VenvPythonw) { return $VenvPythonw }
  $cmd = Get-Command pythonw.exe -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  throw "pythonw.exe not found. Create venv at .venv or install Python with pythonw on PATH."
}

# ---------- Safe getter for HKCU Run value (avoids StrictMode property lookup issues) ----------
function Get-RunKeyValue {
  $v = $null
  try { $v = Get-ItemPropertyValue -Path $RunKeyPath -Name $RunKeyName -ErrorAction SilentlyContinue } catch { }
  return $v
}

# ---------- Create a simple CMD launcher to avoid schtasks.exe quoting issues ----------
function New-LauncherCmd {
  New-Item -Force -ItemType Directory -Path $LauncherDir | Out-Null
  $scriptPath = $PSCommandPath
  $cmdLines = @(
    '@echo off'
    'setlocal'
    'REM Launch the daemon via PowerShell 5.1; working dir handled by _launch'
    '""%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"" -NoProfile -ExecutionPolicy Bypass -File ' +
      '"' + $scriptPath + '" _launch --no-gui'
  )
  Set-Content -Path $LauncherCmd -Value $cmdLines -Encoding ASCII
}

# ---------- Startup shortcut helper ----------
function New-StartupShortcut {
  param(
    [string]$Target,         # usually $LauncherCmd
    [string]$Arguments = '', # leave empty when Target is a .cmd
    [string]$WorkingDir = $env:LOCALAPPDATA
  )
  $shell = New-Object -ComObject WScript.Shell
  $lnk = $shell.CreateShortcut($StartupLnk)
  $lnk.TargetPath   = $Target
  if ($Arguments) { $lnk.Arguments = $Arguments }
  $lnk.WorkingDirectory = $WorkingDir
  $lnk.WindowStyle  = 7  # minimized
  $lnk.IconLocation = "$env:SystemRoot\System32\shell32.dll, 44"
  $lnk.Description  = 'CFB-Tix Daemon (runs at logon)'
  $lnk.Save()
}

# ---------- HKCU Run key helper ----------
function Install-RunKey {
  param(
    [string]$CommandLine  # full command to execute at logon
  )
  New-Item -Path $RunKeyPath -Force | Out-Null
  New-ItemProperty -Path $RunKeyPath -Name $RunKeyName -Value $CommandLine -PropertyType String -Force | Out-Null
}

# ---------- Launch the daemon headlessly ----------
function Invoke-Launch {
  Set-Location -Path $RepoRoot

  # Repo-locked environment + PYTHONPATH
  if ($env:PYTHONPATH) { $env:PYTHONPATH = "$SrcDir;$($env:PYTHONPATH)" } else { $env:PYTHONPATH = "$SrcDir" }
  if (-not $env:REPO_DATA_LOCK)          { $env:REPO_DATA_LOCK = '1' }
  if (-not $env:REPO_ALLOW_NON_REPO_OUT) { $env:REPO_ALLOW_NON_REPO_OUT = '0' }
  if (-not $env:CFB_TIX_ENABLE_SYNC)     { $env:CFB_TIX_ENABLE_SYNC = '0' }  # flip to '1' to auto-commit/push

  $pythonw = Get-Pythonw
  $args    = @('-m','cfb_tix.daemon','run')
  if ($NoGui) { $args += '--no-gui' }

  Write-Note "Launching daemon with: $pythonw $($args -join ' ')"
  Start-Process -FilePath $pythonw -ArgumentList $args -WorkingDirectory $RepoRoot -WindowStyle Hidden | Out-Null
}

# ---------- Install autostart (multi-stage fallbacks) ----------
function Install-Task {
  $currentIdentity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
  $scriptPath = $PSCommandPath
  $arg = "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" _launch --no-gui"

  $action   = $null
  $trigger  = $null
  $settings = $null

  # Only build ScheduledTasks objects if module loaded successfully
  if (Get-Module ScheduledTasks) {
    $action   = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $RepoRoot
    $trigger  = New-ScheduledTaskTrigger -AtLogOn
    $settings = New-ScheduledTaskSettingsSet `
      -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
      -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
      -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 0)

    # Remove existing Scheduled Task (ignore errors)
    try {
      $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
      if ($existing) { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false }
    } catch { }
  }

  $mode = $null  # track which install path succeeded

  # Stage A: ScheduledTasks with explicit principal
  if ($action -and $trigger -and $settings) {
    try {
      $principal = New-ScheduledTaskPrincipal -UserId $currentIdentity -LogonType Interactive
      $task = New-ScheduledTask -Action $action -Trigger $trigger -Settings $settings -Principal $principal
      Register-ScheduledTask -TaskName $TaskName -InputObject $task | Out-Null
      $mode = 'scheduled_task_with_principal'
      Write-Note "Task registered (with principal): $currentIdentity"
    } catch {
      if ($_.Exception.HResult -eq -2147024891) {
        Write-Note "Principal register denied; retrying without principal…"
      } else {
        Write-Note "Principal register failed ($($_.Exception.Message)); retrying without principal…"
      }
    }
  }

  # Stage B: ScheduledTasks without principal
  if (-not $mode -and $action -and $trigger -and $settings) {
    try {
      $task = New-ScheduledTask -Action $action -Trigger $trigger -Settings $settings
      Register-ScheduledTask -TaskName $TaskName -InputObject $task | Out-Null
      $mode = 'scheduled_task_implicit'
      Write-Note "Task registered (implicit current user)."
    } catch {
      Write-Note "Implicit register failed ($($_.Exception.Message)); will try schtasks.exe…"
    }
  }

  # Stage C: schtasks.exe with robust .cmd launcher
  if (-not $mode) {
    New-LauncherCmd
    $trQuoted = '"' + $LauncherCmd + '"'
    $schtasksArgs = @(
      '/Create',
      '/SC','ONLOGON',
      '/TN',"$TaskName",
      '/TR',$trQuoted,
      '/RL','LIMITED',
      '/F'
    )
    & schtasks.exe @schtasksArgs | Out-Null
    if ($LASTEXITCODE -eq 0) {
      # Verify presence
      $check = $null
      try { $check = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue } catch { }
      if ($check) {
        $mode = 'schtasks_cmd'
        Write-Note "Task registered via schtasks.exe fallback."
      } else {
        Write-Note "schtasks.exe reported success but task not found; will try Startup shortcut…"
      }
    } else {
      Write-Note "schtasks.exe failed with exit code $LASTEXITCODE; will try Startup shortcut…"
    }
  }

  # Stage D: Per-user Startup shortcut (.lnk)
  if (-not $mode) {
    New-LauncherCmd
    New-StartupShortcut -Target $LauncherCmd -WorkingDir $RepoRoot
    if (Test-Path $StartupLnk) {
      $mode = 'startup_shortcut'
      Write-Note "Installed Startup shortcut: $StartupLnk"
    } else {
      Write-Note "Failed to create Startup shortcut; will try HKCU Run key…"
    }
  }

  # Stage E: HKCU Run registry (last resort)
  if (-not $mode) {
    New-LauncherCmd
    $cmdLine = '"' + $LauncherCmd + '"'
    Install-RunKey -CommandLine $cmdLine
    # Verify safely (StrictMode-proof)
    $val = Get-RunKeyValue
    if ($val) {
      $mode = 'run_key'
      Write-Note "Installed Run key entry at $RunKeyPath\$RunKeyName"
    } else {
      throw "Failed to register via all methods (policy blocks)."
    }
  }

  Write-Note "✅ Installed/updated autostart via: $mode"
  Write-Note "  Repo root : $RepoRoot"
  Write-Note "  Script    : $scriptPath"
  Write-Note "  Launcher  : $LauncherCmd"
  if ($mode -eq 'startup_shortcut') { Write-Note "  Startup   : $StartupLnk" }
  if ($mode -like 'scheduled_task*' -or $mode -eq 'schtasks_cmd') { Write-Note "  TaskName  : $TaskName" }
  if ($mode -eq 'run_key') { Write-Note "  RunKey    : $RunKeyPath\$RunKeyName" }

  # Start a headless instance NOW so you don't have to log off
  try { Invoke-Launch } catch { Write-Note "Launch now failed: $($_.Exception.Message)" }
}

# ---------- Task controls ----------
function Start-TaskSafe {
  $t = $null
  try { $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue } catch { }
  if (-not $t) { Write-Note "No Scheduled Task to start (using Startup/RunKey mode). Launching one-off…"; Invoke-Launch; return }
  Start-ScheduledTask -TaskName $TaskName
  Write-Note "▶️  Started '$TaskName'."
}

function Stop-TaskSafe {
  $t = $null
  try { $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue } catch { }
  if (-not $t) { Write-Note "No Scheduled Task present. If a background instance is running, use 'kill'."; return }
  try {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Write-Note "⏹️  Stopped '$TaskName'."
  } catch {
    Write-Note "Stop requested; if nothing was running, that's fine."
  }
}

function Uninstall-Task {
  # Remove scheduled task if exists
  try {
    $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($t) { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false; Write-Note "🗑️  Uninstalled Scheduled Task '$TaskName'." }
  } catch { }

  # Remove startup shortcut
  if (Test-Path $StartupLnk) {
    try { Remove-Item -Path $StartupLnk -Force; Write-Note "🗑️  Removed Startup shortcut." } catch { }
  }

  # Remove Run key
  try {
    $existingVal = Get-RunKeyValue
    if ($existingVal) {
      Remove-ItemProperty -Path $RunKeyPath -Name $RunKeyName -ErrorAction SilentlyContinue
      Write-Note "🗑️  Removed Run key entry."
    }
  } catch { }

  Write-Note "Uninstall cleanup complete."
}

# ---------- Status & logs ----------
function Show-Status {
  $task = $null
  try { $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue } catch { }
  $lnk  = Test-Path $StartupLnk
  $rk   = Get-RunKeyValue

  if ($task) {
    $info = $null
    try { $info = Get-ScheduledTaskInfo -TaskName $TaskName } catch { }
    Write-Host "Autostart : Scheduled Task"
    Write-Host "Task      : $TaskName"
    Write-Host "State     : $($task.State)"
    if ($info -and $info.LastRunTime) { Write-Host ("Last run  : {0}" -f $info.LastRunTime.ToLocalTime()) }
    if ($info -and $info.NextRunTime) { Write-Host ("Next run  : {0}" -f $info.NextRunTime.ToLocalTime()) }
  } elseif ($lnk) {
    Write-Host "Autostart : Startup Shortcut"
    Write-Host "Shortcut  : $StartupLnk"
  } elseif ($rk) {
    Write-Host "Autostart : HKCU Run Key"
    Write-Host "RunKey    : $RunKeyPath\$RunKeyName"
    Write-Host "Command   : $rk"
  } else {
    Write-Host "Status    : NOT INSTALLED"
  }
  Write-Host "Log file  : $MainLog"
}

function Tail-Log([int]$n = 200) {
  if (-not (Test-Path $MainLog)) { Write-Host "Log not found yet: $MainLog"; return }
  Write-Host "=== Last $n lines of $MainLog ==="
  Get-Content -Path $MainLog -Tail $n | ForEach-Object { $_ }
}

# ---------- Kill/clean lock ----------
function Kill-FromLock {
  if (-not (Test-Path $LockFile)) { Write-Note "No lock file at $LockFile"; return }
  $pid = $null
  try {
    $pidStr = (Get-Content -Path $LockFile | Select-Object -First 1).Trim()
    if ($pidStr -match '^\d+$') { $pid = [int]$pidStr }
  } catch { }

  if (-not $pid) {
    Write-Note "Lock file exists but PID is missing/invalid → removing stale lock."
    try { Remove-Item -Path $LockFile -Force } catch { }
    return
  }

  try {
    $p = Get-Process -Id $pid -ErrorAction Stop
    Write-Note "Killing process PID $pid ($($p.ProcessName))"
    Stop-Process -Id $pid -Force
  } catch {
    Write-Note "Process PID $pid not found; removing stale lock."
  }

  try { Remove-Item -Path $LockFile -Force; Write-Note "Removed lock file: $LockFile" } catch { }
}

function Clean-Lock {
  if (Test-Path $LockFile) {
    try { Remove-Item -Path $LockFile -Force; Write-Note "Removed lock file: $LockFile" } catch { Write-Note "Failed to remove lock file (will ignore)." }
  } else {
    Write-Note "No lock file to remove."
  }
}

# ---------- Router ----------
switch ($Command) {
  '_launch'  { Invoke-Launch }
  'install'  { Install-Task; try { Start-TaskSafe } catch { Write-Note $_.Exception.Message }; Write-Note "✅ Ready. It will also start at user logon." }
  'start'    { Start-TaskSafe }
  'stop'     { Stop-TaskSafe }
  'status'   { Show-Status }
  'log'      { Tail-Log -n $Lines }
  'runonce'  { Invoke-Launch; Write-Note "✅ Launched one detached instance of the daemon." }
  'kill'     { Kill-FromLock }
  'cleanlock'{ Clean-Lock }
  'uninstall'{ Uninstall-Task }
  default    { throw "Unknown command: $Command" }
}
