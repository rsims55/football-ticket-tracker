import sys, os, site, traceback
OUT = os.path.join(os.path.dirname(__file__), "diag_out.txt")
def line(*a): 
    with open(OUT, "a", encoding="utf-8") as f: 
        print(*a, file=f)

open(OUT, "w").close()
line("sys.executable =", sys.executable)
line("sys.version    =", sys.version)
line("site-packages  =", site.getsitepackages() if hasattr(site,"getsitepackages") else "n/a")
line("sys.path[0:3]  =", sys.path[:3])

try:
    import pandas as pd
    line("pandas OK      =", pd.__version__)
    line("pandas file    =", getattr(pd, "__file__", "n/a"))
except Exception:
    line("pandas ERROR   =")
    line(traceback.format_exc())
