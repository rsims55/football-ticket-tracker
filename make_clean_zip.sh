#!/usr/bin/env bash
set -euo pipefail

# make_current_zip.sh
# Creates/overwrites a single "<repo>-CURRENT.zip" archive of this repo.
# Default OUTDIR is the repo's parent directory (i.e., right next to the repo).
#
# Env flags:
#   OUTDIR=...       where to place the archive (default: parent of repo)
#   INCLUDE_DATA=1   include data/* (default: excluded)
#   INCLUDE_VENV=1   include .venv/, venv/, env/ (default: excluded)
#   VERBOSE=1        show zip's per-file output
#   LIST=0           skip listing archive contents after creation

usage() {
  cat <<'USAGE'
Usage: ./make_current_zip.sh
Env:
  OUTDIR=PATH        Output directory (default: parent of repo)
  INCLUDE_DATA=1     Include data/* (default excluded)
  INCLUDE_VENV=1     Include virtualenvs (default excluded)
  VERBOSE=1          Verbose zip output
  LIST=0             Skip listing archive contents
USAGE
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }

# Resolve repo root robustly (works inside subdirs too)
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

REPO_NAME="$(basename "$REPO_ROOT")"
PARENT_DEFAULT="$(cd "$REPO_ROOT/.." && pwd)"
OUTDIR="${OUTDIR:-$PARENT_DEFAULT}"

mkdir -p "$OUTDIR"

# Pick archiver
if command -v zip >/dev/null 2>&1; then
  ARCH="zip"
elif command -v tar >/dev/null 2>&1; then
  ARCH="tar"
else
  echo "❌ Need either 'zip' or 'tar' installed." >&2
  exit 1
fi

# Exclusions
EXCLUDES=(
  ".git/*"
  "__pycache__/*" "*.pyc" "*.pyo" "*.pyd"
  ".DS_Store" "Thumbs.db"
  "*.log" "logs/*" ".pytest_cache/*" ".coverage" "htmlcov/*" ".mypy_cache/*" ".ruff_cache/*" ".ipynb_checkpoints/*"
  "build/*" "dist/*" "*.egg-info/*" "packaging/dist/*" "packaging/windows/output/*" "*.ext4" "*.exe" "*.msi"
  "*.sqlite*" "*.db"
)

# Virtual envs
if [[ "${INCLUDE_VENV:-0}" != "1" ]]; then
  EXCLUDES+=(".venv/*" "venv/*" "env/*")
fi

# Data (heavy) folders
if [[ "${INCLUDE_DATA:-0}" != "1" ]]; then
  EXCLUDES+=("data/daily/*" "data/predicted/*" "data/tmp/*" "data/raw/*")
fi

BASENAME="${REPO_NAME}-CURRENT"
FINAL_ZIP="$OUTDIR/${BASENAME}.zip"
FINAL_TGZ="$OUTDIR/${BASENAME}.tar.gz"

# Avoid archiving the archive if OUTDIR==REPO_ROOT
SELF_EXCLUDES=()
if [[ "$OUTDIR" == "$REPO_ROOT" ]]; then
  SELF_EXCLUDES+=("${BASENAME}.zip" "${BASENAME}.tar.gz")
fi

# Create to a temp file first, then move atomically
cleanup() { [[ -n "${TMP:-}" && -f "$TMP" ]] && rm -f "$TMP"; }
trap cleanup EXIT

if [[ "$ARCH" == "zip" ]]; then
  TMP="$(mktemp -p "$OUTDIR" "${BASENAME}.XXXXXX.zip")"
  echo "📦 Creating ZIP: $FINAL_ZIP"
  ZIPFLAGS=(-r -9)
  [[ "${VERBOSE:-0}" == "1" ]] && ZIPFLAGS+=(-v) || ZIPFLAGS+=(-q)
  # shellcheck disable=SC2068
  zip "${ZIPFLAGS[@]}" "$TMP" . -x ${EXCLUDES[@]} ${SELF_EXCLUDES[@]}
  mv -f "$TMP" "$FINAL_ZIP"
  echo "✅ Wrote: $FINAL_ZIP"
  OUTFILE="$FINAL_ZIP"
else
  TMP="$(mktemp -p "$OUTDIR" "${BASENAME}.XXXXXX.tar.gz")"
  echo "📦 Creating tar.gz: $FINAL_TGZ"
  TAR_EX=()
  for pat in "${EXCLUDES[@]}" "${SELF_EXCLUDES[@]}"; do TAR_EX+=( "--exclude=$pat" ); done
  tar -czf "$TMP" "${TAR_EX[@]}" .
  mv -f "$TMP" "$FINAL_TGZ"
  echo "✅ Wrote: $FINAL_TGZ"
  OUTFILE="$FINAL_TGZ"
fi

# Optional quick listing
if [[ "${LIST:-1}" == "1" ]]; then
  echo
  echo "👀 First 50 entries in the archive:"
  case "$OUTFILE" in
    *.zip)
      if command -v unzip >/dev/null 2>&1; then
        unzip -l "$OUTFILE" | sed -n '1,50p'
      else
        echo "(install 'unzip' to list contents)"
      fi
      ;;
    *.tar.gz)
      tar -tzf "$OUTFILE" | sed -n '1,50p'
      ;;
  esac
fi
