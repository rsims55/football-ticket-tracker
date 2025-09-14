Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Set-Location 'C:\Users\randi\GitHub Repos\football-ticket-tracker\football-ticket-tracker'
& 'C:\Users\randi\GitHub Repos\football-ticket-tracker\football-ticket-tracker\.venv\Scripts\python.exe' 'src\cfb_tix\daemon.py' *> 'C:\Users\randi\GitHub Repos\football-ticket-tracker\football-ticket-tracker\logs\daemon.log'
