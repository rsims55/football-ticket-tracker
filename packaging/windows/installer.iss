; packaging/windows/installer.iss
#define AppName    "CFB Ticket Price Tracker"
; Stable GUID for upgrades; generate once and keep forever:
#define AppId      "{A3E3E5F2-9A9F-4D1E-BB5B-2A8B7B0C2C11}"
; Allow override: ISCC installer.iss /DAppVersion=1.2.3
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

#define SourceBase "..\.."     ; repo root relative to this .iss file
#define OutputDir  "Output"

[Setup]
AppId={#AppId}
AppName={#AppName}
AppVersion={#AppVersion}
; Per-user install location to match PowerShell ($env:LOCALAPPDATA\cfb-tix)
DefaultDirName={localappdata}\cfb-tix
DefaultGroupName=CFB Tix
DisableDirPage=yes
DisableProgramGroupPage=yes
OutputDir={#OutputDir}
OutputBaseFilename=cfb-tix-setup
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=lowest
UsePreviousAppDir=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
; Optional checkbox to also install the daily CSV sync
Name: "install_sync"; Description: "Set up daily CSV sync at 06:10"; Flags: unchecked

[Files]
; Copy the repo into {app}\app, excluding junk (comma-separated list)
Source: "{#SourceBase}\*"; DestDir: "{app}\app"; Flags: recursesubdirs ignoreversion; \
  Excludes: ".git\*,.github\*,packaging\dist\*,dist\*,build\*,__pycache__\*,*.pyc,*.egg-info\*,.venv\*,.mypy_cache\*,.pytest_cache\*,.DS_Store"

; PowerShell installers (kept canonical in repo)
Source: "{#SourceBase}\packaging\windows\install_win.ps1"; DestDir: "{app}"; Flags: ignoreversion
; If you added this per earlier message:
Source: "{#SourceBase}\packaging\windows\install_sync_win.ps1"; DestDir: "{app}"; Flags: ignoreversion; Check: FileExists(ExpandConstant('{#SourceBase}\packaging\windows\install_sync_win.ps1'))

; Icon (optional)
Source: "{#SourceBase}\assets\icons\cfb-tix.ico"; DestDir: "{app}\assets\icons"; Flags: ignoreversion recursesubdirs createallsubdirs

[Run]
; Install app for current user (matches your install_win.ps1 semantics)
Filename: "powershell.exe"; \
  Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install_win.ps1"" -AppDir ""{app}"""; \
  Flags: nowait postinstall skipifsilent

; Optional: install daily sync if the task was selected
Filename: "powershell.exe"; \
  Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install_sync_win.ps1"" -RepoDir ""{app}\app"" -RunTime ""06:10"""; \
  Flags: nowait postinstall skipifsilent; \
  Check: WizardIsTaskSelected('install_sync') and FileExists(ExpandConstant('{app}\install_sync_win.ps1'))

[UninstallRun]
; Uninstall app tasks/shortcuts/payload (your PS script already handles main tasks)
Filename: "powershell.exe"; \
  Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install_win.ps1"" -Uninstall"; \
  RunOnceId: "CFB-Tix-Uninstall"; Flags: skipifsilent

; Also clean up the daily sync task if it was created
Filename: "schtasks.exe"; \
  Parameters: "/Delete /TN ""cfb-tix-sync"" /F"; \
  RunOnceId: "CFB-Tix-Uninstall-Sync"; Flags: skipifsilent

[Icons]
; Your PowerShell creates the Start Menu shortcut after the venv exists, so nothing required here.
; If you later want an extra doc link, add it here under the "CFB Tix" group.
