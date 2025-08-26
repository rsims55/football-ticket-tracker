import os, sys
from pathlib import Path
ROOT = Path(r"C:\Users\randi\GitHub Repos\football-ticket-tracker\football-ticket-tracker")
sys.path.insert(0, str(ROOT / "src"))
from cfb_tix import daemon as d
p = d.detect_paths()
d.do_sync(p, "manual_final_sync")
print("✅ final sync done")
