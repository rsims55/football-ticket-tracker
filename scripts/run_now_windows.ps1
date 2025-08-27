# scripts/run_now_windows.ps1
# Run the pipeline with .venv and print proofs (hashes, row counts).
# You can run from anywhere: 
#   powershell -ExecutionPolicy Bypass -File "C:\Users\randi\GitHub Repos\football-ticket-tracker\football-ticket-tracker\scripts\run_now_windows.ps1"

$ErrorActionPreference = "Stop"

# --- Repo paths (absolute) ---
$Repo   = "C:\Users\randi\GitHub Repos\football-ticket-tracker\football-ticket-tracker"
$Py     = "$Repo\.venv\Scripts\python.exe"
$CSV    = "$Repo\data\daily\price_snapshots.csv"
$PKL    = "$Repo\models\ticket_price_model.pkl"
$PRED   = "$Repo\data\predicted\predicted_prices_optimal.csv"
$Script = "$Repo\run_pipeline_now.py"

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

# Safety lock
$NoSync = Join-Path $Repo ".cfb_tix.NOSYNC"
if (-not (Test-Path $NoSync)) { New-Item -ItemType File -Path $NoSync -Force | Out-Null }

# --- Run orchestrator directly by absolute path ---
Write-Host "`n=== RUN: $Script ==="
& $Py $Script
$code = $LASTEXITCODE
if ($code -ne 0) { throw "run_pipeline_now.py failed with exit code $code" }

Start-Sleep -Milliseconds 300

Write-Host "`n=== AFTER ==="
$csvHash1  = Sha256 $CSV;  "Snapshots SHA256: $csvHash1"
$predHash1 = Sha256 $PRED; "Predicted SHA256: $predHash1"
$csvRows1  = CsvRows $CSV; "Snapshot rows:    $csvRows1"
FileMeta $PKL | Out-Host
FileMeta $PRED | Out-Host

# Verdicts
$csvDelta = ($csvRows1 - $csvRows0)
if ($csvHash1 -ne $csvHash0 -and $csvDelta -ge 1) {
  Write-Host "SNAPSHOTS UPDATED (rows added: $csvDelta)"
} else {
  Write-Warning "SNAPSHOTS DID NOT CHANGE"
}

if ($predHash1 -ne $predHash0) {
  Write-Host "PREDICTIONS UPDATED"
} else {
  Write-Warning "PREDICTIONS DID NOT CHANGE"
}

Write-Host ""
Write-Host "=== Git status (no auto-commit expected) ==="
git -C $Repo status --porcelain=v1
