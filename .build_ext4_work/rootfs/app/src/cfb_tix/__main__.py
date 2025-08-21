# src/cfb_tix/__main__.py
from .daemon import main
import sys
main(no_gui=("--no-gui" in sys.argv))
