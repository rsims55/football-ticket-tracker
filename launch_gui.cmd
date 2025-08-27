@echo off
setlocal
set "PYTHONPATH=C:\Users\randi\GitHub Repos\football-ticket-tracker\football-ticket-tracker\src"
set "REPO_DATA_LOCK=1"
"C:\Users\randi\GitHub Repos\football-ticket-tracker\football-ticket-tracker\.venv\Scripts\pythonw.exe" -m gui.ticket_predictor_gui
endlocal
