<#  reset_windows.ps1
    Fully reset the cfb-tix Windows install:
    - Stops & removes scheduled tasks
    - Kills any running cfb-tix python processes
    - Deletes local app payload
    - Removes Start Menu / Desktop shortcuts
    - Cleans repo build artifacts (if run inside the repo)
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'SilentlyContinue'

# --- Names & paths (keep in sync with your installer) ---
$AppName       = 'cfb-tix'
$Tasks         = @('cfb-tix-sync','cfb-tix-upload','cfb-tix-gui')  # harmless if some don't exist
$AppDir        = Join-Path $env:LOCALAPPDATA $AppName               # e.g. C:\Users\you\AppData\Local\cfb-tix
$StartMenuDir  = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\CFB Tix'
$DesktopLink   = Join-Path $env:USERPROFILE 'Desktop\CFB Tix GUI.lnk'

Write-Host "→ Stopping & removing Scheduled Tasks (if present)…"
foreach ($t in $Tasks) {
  $task = Get-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue
  if ($task) {
    Stop-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $t -Confirm:$false -ErrorAction SilentlyContinue
  }
}

Write-Host "→ Killing any running cfb-tix processes…"
try {
  Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -and ($_.CommandLine -match 'cfb[-_]tix') } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
} catch {}

Write-Host "→ Removing local install payload…"
if (Test-Path $AppDir) { Remove-Item -Recurse -Force $AppDir }

Write-Host "→ Removing Start Menu shortcuts…"
if (Test-Path $StartMenuDir) { Remove-Item -Recurse -Force $StartMenuDir }

Write-Host "→ Removing Desktop shortcut…"
if (Test-Path $DesktopLink) { Remove-Item -Force $DesktopLink }

# Optional: clean user-wide temp cache the installer might have used
#$TempDir = Join-Path $env:TEMP 'cfb-tix'
#if (Test-Path $TempDir) { Remove-Item -Recurse -Force $TempDir }

Write-Host "→ Cleaning repo build artifacts (if present)…"
$repoArtifacts = @(
  'dist', '.build_ext4_work', 'build', 'packaging\dist',
  (Get-ChildItem -Name -Filter '*.egg-info' -ErrorAction SilentlyContinue)
) | Where-Object { $_ -and (Test-Path $_) }
foreach ($p in $repoArtifacts) { Remove-Item -Recurse -Force $p }

Write-Host "✅ Reset complete."
