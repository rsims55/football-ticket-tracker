; packaging/windows/installer.iss
#define AppName "CFB Ticket Price Tracker"
#define AppId   "CFB-Tix-Tracker"
; AppVersion can be overridden from command line: ISCC installer.iss /DAppVersion=1.2.3
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

#define SourceBase "..\.."     ; repo root relative to this .iss file
#define OutputDir  "Output"

[Setup]
AppId={{#AppId}
AppName={#AppName}
AppVersion={#AppVersion}
DefaultDirName={autopf}\CFB Ticket Price Tracker
DefaultGroupName=CFB Ticket Price Tracker
DisableDirPage=yes
DisableProgramGroupPage=yes
OutputDir={#OutputDir}
OutputBaseFilename=cfb-tix-setup
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=lowest
; Do not require admin; we install per-user stuff with Task Scheduler.

[Files]
; Copy the whole repo into {app}\app excluding junk
Source: "{#SourceBase}\*"; DestDir: "{app}\app"; Flags: recursesubdirs ignoreversion; Excludes: ".git\*;dist\*;build\*;__pycache__\*;*.pyc;.github\*;*.egg-info\*"
; PowerShell installer
Source: "{#SourceBase}\packaging\windows\install_win.ps1"; DestDir: "{app}"; Flags: ignoreversion
; Icon
Source: "{#SourceBase}\assets\icons\cfb-tix.ico"; DestDir: "{app}\assets\icons"; Flags: ignoreversion recursesubdirs createallsubdirs

[Run]
Filename: "powershell.exe";
Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\packaging\windows\register_sync.ps1"" -Repo ""{app}""";
Flags: runhidden nowait postinstall skipifsilent

[UninstallRun]
; On uninstall, remove scheduled task and shortcut (idempotent)
Filename: "powershell.exe"; \
  Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install_win.ps1"" -Uninstall"; \
  RunOnceId: "CFB-Tix-Uninstall"; Flags: skipifsilent

[Icons]
; (Optional) Add a doc link or uninstaller icon in group; main GUI shortcut is created by PowerShell after venv exists

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
