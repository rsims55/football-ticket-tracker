param(
  [string]$RepoDir   = "$env:LOCALAPPDATA\cfb-tix\app",
  [string]$PythonBin = "$env:LOCALAPPDATA\cfb-tix\venv\Scripts\python.exe",
  [string]$At        = "06:10",
  [switch]$Unregister
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$TaskName = "cfb-tix-sync"
$EnvFile  = Join-Path $env:APPDATA "cfb-tix\env"
$EnvDir   = Split-Path $EnvFile -Parent
$BinDir   = Join-Path $env:LOCALAPPDATA "cfb-tix\bin"
$Runner   = Join-Path $BinDir "run_sync.ps1"

function Ensure-Time([string]$hhmm) {
  if (-not [System.Text.RegularExpressions.Regex]::IsMatch($hhmm, '^(?:[01]\d|2[0-3]):[0-5]\d$')) {
    throw "Invalid -At time '$hhmm'. Use HH:mm (e.g., 06:10)."
  }
  [DateTime]::ParseExact($hhmm,'HH:mm',$null)
}

# --- Unregister mode ---
if ($Unregister) {
  try { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop } catch { }
  Write-Host "✅ Unregistered '$TaskName'."
  return
}

# --- Resolve Python ---
if (-not (Test-Path $PythonBin)) {
  $py = (Get-Command python -ErrorAction SilentlyContinue)?.Source
  if (-not $py) { $py = (Get-Command py -ErrorAction SilentlyContinue)?.Source }
  if (-not $py) { throw "Python not found. Set -PythonBin or add python/py to PATH." }
  $PythonBin = $py
}

# --- Validate / create dirs ---
New-Item -ItemType Directory -Force -Path $EnvDir, $BinDir | Out-Null
if (-not (Test-Path $RepoDir)) { throw "RepoDir not found: $RepoDir" }

# --- Prompt once for GH token (stored as GH_TOKEN in %APPDATA%\cfb-tix\env) ---
if (-not (Test-Path $EnvFile) -or -not (Select-String -Path $EnvFile -SimpleMatch "GH_TOKEN=" -ErrorAction SilentlyContinue)) {
  Write-Host "🔑 No GitHub token found. (Press Enter to skip uploads.)"
  $sec = Read-Host -AsSecureString "Paste your GitHub access token"
  $plain = if ($sec.Length -gt 0) {
    [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec))
  } else { "" }
  "GH_TOKEN=$plain" | Out-File -FilePath $EnvFile -Encoding UTF8 -Force
  Write-Host ("✅ Saved token to {0}" -f $EnvFile)
}

# --- Wrapper that loads GH_TOKEN then runs the sync ---
@"
`$ErrorActionPreference = 'Stop'
if (Test-Path '$EnvFile') {
  Get-Content '$EnvFile' | ForEach-Object {
    if (`$_ -match '=') { `$name,`$val = `$_ -split '=',2; [Environment]::SetEnvironmentVariable(`$name, `$val, 'Process') }
  }
}
& '$PythonBin' '$RepoDir\scripts\sync_snapshots.py' pull_push
"@ | Out-File -FilePath $Runner -Encoding UTF8 -Force

# --- First-time pull (download only; non-fatal) ---
try {
  Push-Location $RepoDir
  & $PythonBin "$RepoDir\scripts\sync_snapshots.py" pull 2>$null
} catch { } finally { Pop-Location }

# --- Replace scheduled task (daily at HH:mm, Limited) ---
$time = Ensure-Time $At
try { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue } catch { }

$action    = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Runner`""
$trigger   = New-ScheduledTaskTrigger -Daily -At $time
$settings  = New-ScheduledTaskSettingsSet -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -LogonType S4U -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null

Write-Host "✅ Registered '$TaskName' at $At daily."
Write-Host "   Repo: $RepoDir"
Write-Host "   Token file: $EnvFile"
