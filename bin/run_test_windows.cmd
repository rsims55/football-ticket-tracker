@echo off
rem Test run: scrapes only 3 random teams instead of the full list.
rem All other pipeline steps run normally.
setlocal
set ROOT_DIR=%~dp0\..
cd /d %ROOT_DIR%

rem Bootstrap venv if missing or empty
if not exist ".venv\Scripts\activate" (
  echo Creating .venv...
  python -m venv .venv
)
call .venv\Scripts\activate
python -c "import pandas" >nul 2>&1
if errorlevel 1 (
  echo Installing dependencies...
  pip install --upgrade pip -q
  pip install -r requirements.txt -q
)

set TEAMS_LIMIT=3

python src\builders\annual_setup.py
python src\builders\weekly_update.py
python src\builders\daily_snapshot.py
python src\modeling\train_catboost_min.py
python src\reports\generate_weekly_report.py
python scripts\health_check.py

endlocal
