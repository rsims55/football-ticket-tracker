#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Bootstrap venv if missing or empty
if [ ! -f ".venv/bin/activate" ]; then
  echo "Creating .venv..."
  python3 -m venv .venv
fi
source .venv/bin/activate
if ! python -c "import pandas" &>/dev/null; then
  echo "Installing dependencies..."
  pip install --upgrade pip -q
  pip install -r requirements.txt -q
fi

python src/builders/annual_setup.py
python src/builders/weekly_update.py
python src/builders/daily_snapshot.py
python src/modeling/train_catboost_min.py
python src/reports/generate_weekly_report.py
python scripts/health_check.py
