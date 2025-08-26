import os, sys, runpy, traceback, ctypes
REPO = r'C:\Users\randi\GitHub Repos\football-ticket-tracker\football-ticket-tracker'
SRC  = os.path.join(REPO, 'src')
VENV_SITE = r'C:\Users\randi\GitHub Repos\football-ticket-tracker\football-ticket-tracker\.venv\Lib\site-packages'
LOG  = os.path.join(REPO, 'gui_error.log')
GUI  = os.path.join(SRC, 'gui', 'ticket_predictor_gui.py')

# Clear any stale error log from previous runs
try:
    if os.path.exists(LOG):
        os.remove(LOG)
except Exception:
    pass

try:
    # Make venv site-packages + repo/src importable and set CWD to repo
    for p in (VENV_SITE, REPO, SRC):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(REPO)

    # Quick probes (helpful if something is missing)
    import PyQt5  # noqa: F401
    import pandas # noqa: F401

    # Launch the GUI script as __main__
    if not os.path.isfile(GUI):
        raise FileNotFoundError(f'GUI script not found: {GUI}')
    runpy.run_path(GUI, run_name='__main__')

except SystemExit:
    raise
except Exception:
    try:
        with open(LOG, 'w', encoding='utf-8') as f:
            f.write('sys.executable = ' + sys.executable + '\n')
            f.write('sys.path (head)= ' + repr(sys.path[:10]) + '\n\n')
            f.write(traceback.format_exc())
        ctypes.windll.user32.MessageBoxW(0,
            f"CFB Tickets GUI crashed. See log:\n{LOG}",
            "CFB Tickets GUI",
            0x00000010 | 0x00040000)  # MB_ICONERROR | MB_TOPMOST
    except Exception:
        pass
    raise
