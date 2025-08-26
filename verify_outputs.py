import os, sys, time
from pathlib import Path

REPO = Path(r"C:\Users\randi\GitHub Repos\football-ticket-tracker\football-ticket-tracker")
SRC  = REPO / "src"
sys.path.insert(0, str(SRC))

from cfb_tix import daemon as d

paths = d.detect_paths()
print("repo_root =", paths.repo_root)
print("cwd       =", os.getcwd())
print("python    =", sys.executable)

pred_csv = paths.app_root / "data" / "predicted" / "predicted_prices_optimal.csv"
model_pkl = paths.app_root / "models" / "ticket_price_model.pkl"
daily_dir = paths.app_root / "data" / "daily"

def show_status():
    print("\n-- STATUS --")
    for p in [pred_csv, model_pkl]:
        print(f"{p} ->", "EXISTS" if p.exists() else "MISSING")
    # show recent files in data/
    print("\nRecent in data/:")
    files = sorted((paths.app_root/"data").rglob("*"), key=lambda p:p.stat().st_mtime if p.is_file() else 0, reverse=True)
    for p in files[:10]:
        try:
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)), p)
        except Exception:
            pass

show_status()

# If missing, force-run jobs via daemon wrappers (they also commit/push)
missing_any = (not pred_csv.exists()) or (not model_pkl.exists())
if missing_any:
    print("\n-- FORCING RUNS --")
    d.job_weekly_update(paths)
    d.job_daily_snapshot(paths)
    d.job_train_model(paths)
    d.job_predict_price(paths)
    print("-- RECHECK --")
    show_status()
