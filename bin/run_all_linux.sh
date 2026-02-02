#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source .venv/bin/activate

python src/builders/annual_setup.py
python src/builders/weekly_update.py
python src/builders/daily_snapshot.py
python src/modeling/train_catboost_min.py
SEASON_YEAR=2025 python src/reports/generate_weekly_report.py
