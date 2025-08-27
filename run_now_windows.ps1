# scripts/run_now_windows.ps1
# Runs your orchestrator with the repo's .venv and prints proofs that files updated.
# Usage (PowerShell):  powershell -ExecutionPolicy Bypass -File .\scripts\run_now_windows.ps1

$ErrorActionPreference = "Stop"

# --- Paths ---
$Repo   = "C:\Users\randi\GitHub Repos\football-ticket-tracker\football-ticket-tracker"
$Py     = Join-Path $Repo ".venv\Scripts\python.exe"
$CSV    = Join-Path $Repo "data\daily\price_snapshots.csv"
$PKL    = Join-Path $Repo "models\ticket_price_model.pkl"
$PRED   = Join-Path $Repo "data\predicted\predicted_prices_optimal.csv"

if (-not (Test-Path $Py)) { throw "Missing .venv Python at: $Py" }

# --- Helpers ---
function FileMeta { param([string]$Path) if (Test-Path $Path) { Get-Item -LiteralPath $Path | Select-Object FullName,Length,LastWriteTime } }
function Sha256  { param([string]$Path) if (Test-Path $Path) { (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash } }
function CsvRows { param([string]$Path) if (Test-Path $Path) { try { (Import-Csv -LiteralPath $Path).Count } catch { $null } } }

Write-Host "=== BEFORE ==="
$csvHash0  = Sha256 $CSV;  "Snapshots SHA256: $csvHash0"
$predHash0 = Sha256 $PRED; "Predicted SHA256: $predHash0"
$csvRows0  = CsvRows $CSV; "Snapshot rows:    $csvRows0"
FileMeta $PKL | Out-Host
FileMeta $PRED | Out-Host

# Safety: ensure NOSYNC lock is present so nothing tries to auto-sync during run
$NoSync = Join-Path $Repo ".cfb_tix.NOSYNC"
if (-not (Test-Path $NoSync)) { New-Item -ItemType File -Path $NoSync -Force | Out-Null }

# --- Run the orchestrator (weekly -> daily -> train -> predict) ---
Push-Location $Repo
Write-Host "`n=== RUN: run_pipeline_now.py (.venv) ==="
& $Py "run_pipeline_now.py"
$code = $LASTEXITCODE
Pop-Location
if ($code -ne 0) { throw "run_pipeline_now.py failed with exit code $code" }

Start-Sleep -Milliseconds 300

Write-Host "`n=== AFTER ==="
$csvHash1  = Sha256 $CSV;  "Snapshots SHA256: $csvHash1"
$predHash1 = Sha256 $PRED; "Predicted SHA256: $predHash1"
$csvRows1  = CsvRows $CSV; "Snapshot rows:    $csvRows1"
FileMeta $PKL | Out-Host
FileMeta $PRED | Out-Host

# --- Verdicts ---
$csvDelta = ($csvRows1 - $csvRows0)
if ($csvHash1 -ne $csvHash0 -and $csvDelta -ge 1) { Write-Host "✅ snapshots updated (Δrows=$csvDelta)" } else { Write-Warning "❌ snapshots did not change" }
if ($predHash1 -ne $predHash0) { Write-Host "✅ predictions updated" } else { Write-Warning "❌ predictions did not change" }

Write-Host "`n=== Git status (no auto-commit expected) ==="
git -C $Repo status --porcelain=v1
