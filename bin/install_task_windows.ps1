# bin/install_task_windows.ps1
# Registers the CFB Ticket Tracker daemon as a Windows Task Scheduler task.
# Runs automatically at logon. No admin rights needed.

$Root      = Split-Path $PSScriptRoot -Parent
$Script    = Join-Path $Root "bin\run_daemon_windows.ps1"
$Python    = Join-Path $Root ".venv_win\Scripts\python.exe"
$TaskName  = "CFBTicketTracker-Daemon"

if (-not (Test-Path $Python)) {
    Write-Error "Python not found. Run .\bin\setup_windows.ps1 first."
    exit 1
}

# Action: launch our wrapper script (which sets PYTHONPATH and runs the daemon)
$Action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NonInteractive -WindowStyle Hidden -File `"$Script`"" `
    -WorkingDirectory $Root

# Trigger: at logon of the current user
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

# Settings: no time limit, restart up to 3 times on failure
$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit 0 `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 2) `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew

# Run as current user (interactive, no elevation)
$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName  $TaskName `
    -Action    $Action `
    -Trigger   $Trigger `
    -Settings  $Settings `
    -Principal $Principal `
    -Description "CFB Ticket Tracker daemon — price scraping and modeling pipeline" `
    -Force | Out-Null

Write-Host "Task '$TaskName' registered."
Write-Host ""
Write-Host "Starting daemon now..."
Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 3

$state = (Get-ScheduledTask -TaskName $TaskName).State
Write-Host "Task state: $state"
Write-Host ""
Write-Host "Useful commands:"
Write-Host "  Check status : Get-ScheduledTask -TaskName '$TaskName'"
Write-Host "  Stop daemon  : Stop-ScheduledTask  -TaskName '$TaskName'"
Write-Host "  Remove task  : Unregister-ScheduledTask -TaskName CFBTicketTracker-Daemon -Confirm:false"
