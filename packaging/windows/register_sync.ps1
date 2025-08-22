param(
  [string]$Repo = "$HOME\cfb-ticket-tracker",
  [string]$At   = "06:10"
)

# Resolve python
$py = (Get-Command python -ErrorAction SilentlyContinue)?.Source
if (-not $py) {
  Write-Host "python.exe not found in PATH. Start a shell with your venv active or set PY explicitly."
  exit 1
}

# Ensure first-time pull (download only)
Push-Location $Repo
& $py "scripts/sync_snapshots.py" "pull" 2>$null
Pop-Location

# Register daily task
$taskName = "CFB-Tix Snapshot Sync"
$action   = New-ScheduledTaskAction -Execute $py -Argument "`"$Repo\scripts\sync_snapshots.py`" pull_push"
$trigger  = New-ScheduledTaskTrigger -Daily -At ([DateTime]::Parse($At))
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -LogonType S4U -RunLevel Limited

# Note: The script reads GH_TOKEN from $Repo\.env if present
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null

Write-Host "✅ Registered '$taskName' at $At daily."
Write-Host "Tip: Put GH_TOKEN=... in $Repo\.env to enable uploads."
