# run_jobs_now.py
import sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cfb_tix import daemon as d

p = d.detect_paths()
print("weekly_update");        d.job_weekly_update(p)
print("daily_snapshot");       d.job_daily_snapshot(p)
print("train_model");          d.job_train_model(p)
print("predict_price");        d.job_predict_price(p)
print("evaluate_predictions"); d.job_evaluate_predictions(p)
print("done")
