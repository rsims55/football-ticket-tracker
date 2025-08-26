import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from cfb_tix import daemon as d

paths = d.detect_paths()
print("==> weekly_update"); sys.stdout.flush()
d.job_weekly_update(paths)

print("==> daily_snapshot"); sys.stdout.flush()
d.job_daily_snapshot(paths)

print("==> train_price_model"); sys.stdout.flush()
d.job_train_model(paths)

print("==> predict_price"); sys.stdout.flush()
d.job_predict_price(paths)

print("==> ALL DONE"); sys.stdout.flush()
