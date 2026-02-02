@echo off
setlocal
set ROOT_DIR=%~dp0\..
cd /d %ROOT_DIR%
call .venv\Scripts\activate
python src\gui\ticket_predictor_gui.py
endlocal
