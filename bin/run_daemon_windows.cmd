@echo off
setlocal
set ROOT=%~dp0\..
cd /d %ROOT%
call .venv\Scripts\activate
python scripts\daemon_loop.py
