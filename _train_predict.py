import sys
from pathlib import Path
ROOT = Path(r"C:\Users\randi\GitHub Repos\football-ticket-tracker\football-ticket-tracker")
sys.path.insert(0, str(ROOT / "src"))
from cfb_tix import daemon as d
p = d.detect_paths()
d.job_train_model(p)
d.job_predict_price(p)
