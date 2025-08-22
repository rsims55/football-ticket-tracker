@echo off
setlocal
set PS=powershell.exe -NoProfile -ExecutionPolicy Bypass

%PS% -File "packaging\windows\install_win.ps1" -AppDir "%CD%"
if errorlevel 1 (
  echo Install failed. See console for errors.
  pause
  exit /b 1
)

rem Optional: set up the daily CSV sync at 06:10 (press Enter to skip token)
%PS% -File "scripts\register_sync.ps1" -At "06:10"

rem Optional: launch GUI immediately
if exist "%LOCALAPPDATA%\cfb-tix\venv\Scripts\cfb-tix-gui.exe" (
  start "" "%LOCALAPPDATA%\cfb-tix\venv\Scripts\cfb-tix-gui.exe"
)

echo Done.
pause
