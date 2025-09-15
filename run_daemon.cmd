@echo off
setlocal
cd /d "C:\Users\randi\GitHub Repos\football-ticket-tracker\football-ticket-tracker"
set PYTHONUNBUFFERED=1
"C:\Users\randi\GitHub Repos\football-ticket-tracker\football-ticket-tracker\.venv\Scripts\python.exe" -u "C:\Users\randi\GitHub Repos\football-ticket-tracker\football-ticket-tracker\src\cfb_tix\daemon.py" >> "C:\Users\randi\GitHub Repos\football-ticket-tracker\football-ticket-tracker\logs\daemon.log" 2>&1
