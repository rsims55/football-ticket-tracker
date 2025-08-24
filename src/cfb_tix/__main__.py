# src/cfb_tix/__main__.py
"""
Module launcher so `python -m cfb_tix ...` behaves like the CLI.
Examples:
  python -m cfb_tix run --no-gui
  python -m cfb_tix run --with-gui
"""
from .daemon import main

if __name__ == "__main__":
    main()
