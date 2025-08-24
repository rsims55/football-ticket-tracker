param(
  [string]$Repo = "$HOME\cfb-ticket-tracker"
)

# Resolve paths
$venv       = Join-Path $Repo 'venv'
$py         = Join-Path $venv 'Scripts\python.exe'
$pyw        = Join-Path $venv 'Scripts\pythonw.exe'
$shortcuts  = Join-Path $venv 'Scripts\cfb-tix-shortcuts.exe'

# Basic checks (best-effort)
if (-not (Test-Path $py))  { Write-Host "⚠️  Missing python.exe at $py";  }
if (-not (Test-Path $pyw)) { Write-Host "⚠️  Missing pythonw.exe at $pyw"; }
if (-not (Test-Path $shortcuts)) { Write-Host "⚠️  Missing cfb-tix-shortcuts.exe at $shortcuts"; }

Push-Location $Repo

# 1) Create Start Menu + Desktop + Startup shortcuts (no console windows)
try {
  & $shortcuts
  Write-Host "✅ Shortcuts created."
} catch {
  Write-Host "⚠️  Shortcut creation failed: $($_.Exception.Message)"
}

# 2) Prompt for GitHub token (GUI), saved to $Repo\.env if provided
try {
  & $pyw -m cfb_tix.windows.data_sync ensure_token
  Write-Host "ℹ️  Token prompt completed."
} catch {
  Write-Host "⚠️  Token prompt failed: $($_.Exception.Message)"
}

# 3) Pull latest snapshots, then upload local file (if token present)
#    This ensures everyone gets the freshest CSV AND contributors can push updates.
try {
  & $py -m cfb_tix.windows.data_sync pull_push
  Write-Host "✅ pull→push completed."
} catch {
  Write-Host "⚠️  pull→push failed: $($_.Exception.Message)"
}

# 4) Remove any old Scheduled Task (we use Startup shortcut now)
try {
  $taskName = "CFB-Tix Snapshot Sync"
  Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue | Out-Null
  Write-Host "🧹 Removed legacy scheduled task (if present)."
} catch {
  Write-Host "ℹ️  No legacy task to remove."
}

Pop-Location

Write-Host "🎉 Windows post-install sync finished."
