#!/usr/bin/env bash
set -euo pipefail

# ---- where am I / what to name the zip ----
ROOT="$(pwd)"
BASENAME="$(basename "$ROOT" | tr ' ' '-')"
TS="$(date +%Y%m%d-%H%M%S)"
OUT="../${BASENAME}-${TS}.zip"

# ---- exclusions (folders/files we don't want in the zip) ----
# add/remove as you like
EXCLUDES=(
  ".git/"
  "venv/"
  ".venv/"
  "node_modules/"
  "__pycache__/"
  "dist/"
  "build/"
  "logs/"
  ".vscode/"
  ".idea/"
  "*.pyc"
  "*.pyo"
  ".DS_Store"
)

echo "📦 Creating: $OUT"
echo "📂 From    : $ROOT"
echo

have() { command -v "$1" >/dev/null 2>&1; }

# Build exclusion flags for 'zip'
_zip_exclude_flags=()
for pat in "${EXCLUDES[@]}"; do
  # zip uses -x with patterns; ensure they match anywhere
  _zip_exclude_flags+=("-x" "$pat" "-x" "*/$pat")
done

if have zip; then
  echo "➡️  Using 'zip' CLI"
  # -r recursive, -9 max compression
  # Note: we zip the *contents* of the current dir (.) to avoid nested dir level
  zip -r -9 "$OUT" . "${_zip_exclude_flags[@]}"
  echo "✅ Done: $OUT"
  exit 0
fi

# ---- Fallback: PowerShell Compress-Archive (Git Bash on Windows) ----
# We'll collect a file list that excludes the patterns, then pass to Compress-Archive
if have powershell.exe || have pwsh; then
  echo "ℹ️  'zip' not found; falling back to PowerShell Compress-Archive"
  # Join patterns into a PowerShell array and filter files using -notlike pattern checks
  # NOTE: This may be a bit slower than 'zip' for huge repos, but works out of the box on Windows.
  PS_BIN="$(command -v pwsh || command -v powershell.exe)"
  # Escape OUT for PowerShell
  OUT_WIN="$(printf '%s\n' "$OUT" | sed 's#/#\\#g')"

  # Build a PowerShell snippet that:
  #  1) collects files recursively
  #  2) excludes anything whose path contains any of the EXCLUDES patterns
  #  3) compresses the remaining files
  PS_SCRIPT='
    $ErrorActionPreference = "Stop"
    $dest  = [System.IO.Path]::GetFullPath("'"$OUT_WIN"'")
    $root  = Get-Location
    $ex    = @('"$(printf "'%s'," "${EXCLUDES[@]}" | sed 's/,$//')"')
    # Normalize to full paths for matching
    $files = Get-ChildItem -Recurse -File -Force | Where-Object {
      $p = $_.FullName
      -not ($ex | ForEach-Object { $p -like "*$_*" } | Where-Object { $_ })
    }
    if (Test-Path -LiteralPath $dest) { Remove-Item -LiteralPath $dest -Force }
    if ($files.Count -eq 0) {
      Write-Host "No files to archive after exclusions."
      exit 0
    }
    Compress-Archive -Path ($files | ForEach-Object { $_.FullName }) -DestinationPath $dest -CompressionLevel Optimal -Force
    Write-Host "Compressed $($files.Count) files to $dest"
  '
  "$PS_BIN" -NoProfile -Command "$PS_SCRIPT"
  echo "✅ Done: $OUT"
  exit 0
fi

echo "❌ Neither 'zip' nor PowerShell found. Install one of:"
echo "   - zip (recommended), or"
echo "   - PowerShell 5+/7+"
exit 1
