# src/cfb_tix/launchers.py
"""
GUI-style entry points so Windows launches with pythonw (no console window).
Used by [project.gui-scripts] -> CFB-Ticket-Tracker in pyproject.toml.
"""

import sys


def run_daemon_silently():
    """
    Start the daemon in background mode (no GUI) without opening a console window.
    Equivalent to: cfb-tix run --no-gui
    """
    from . import daemon
    # Pretend CLI args came in this way
    sys.argv = ["cfb-tix", "run", "--no-gui"]
    daemon.main()


if __name__ == "__main__":
    # Allow manual testing: `python -m cfb_tix.launchers`
    run_daemon_silently()
