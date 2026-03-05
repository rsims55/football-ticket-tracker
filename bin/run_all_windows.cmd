@echo off
setlocal
set ROOT_DIR=%~dp0\..
cd /d %ROOT_DIR%
call .venv\Scripts\activate

python src\builders\annual_setup.py
python src\builders\weekly_update.py
python src\builders\daily_snapshot.py
python src\modeling\train_catboost_min.py
python src\reports\generate_weekly_report.py
python scripts\health_check.py

endlocal
