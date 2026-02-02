@echo off
setlocal
set ROOT_DIR=%~dp0\..
cd /d %ROOT_DIR%

if not exist requirements.txt (
  echo requirements.txt not found. Run setup from repo root.
  exit /b 1
)

python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo Setup complete. Activate with: .venv\Scripts\activate
endlocal
