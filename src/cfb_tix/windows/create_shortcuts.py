# src/cfb_tix/windows/create_shortcuts.py
"""
Creates Start Menu + Desktop (GUI) shortcuts and a Startup (daemon) shortcut
with proper icons, launching WITHOUT any console window.

Usage (inside your venv after `pip install -e .`):
    cfb-tix-shortcuts
"""
import os
import sys
import textwrap
from pathlib import Path

try:
    from importlib import resources as importlib_resources
except Exception:  # pragma: no cover
    import importlib_resources  # type: ignore

APP_DIR_NAME = "CFB Ticket Tracker"


def _scripts_dir() -> Path:
    # venv\Scripts on Windows (bin on POSIX). We only use on Windows.
    return Path(sys.executable).parent


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _icons_dir():
    # packaged icons live inside the cfb_tix package
    return importlib_resources.files("cfb_tix").joinpath("assets", "icons")


def _pick_icon(preferred_names):
    icons_root = _icons_dir()
    for name in preferred_names:
        cand = icons_root.joinpath(name)
        try:
            if cand.is_file():
                return str(cand)
        except Exception:
            pass
    # fallback: first .ico found
    try:
        for entry in icons_root.iterdir():
            if entry.name.lower().endswith(".ico"):
                return str(entry)
    except Exception:
        pass
    raise FileNotFoundError(
        "No .ico icons found in package at cfb_tix/assets/icons. "
        "Add your icons to src/cfb_tix/assets/icons/"
    )


def _detect_working_dir() -> str:
    """
    Where shortcuts should start (cwd).
    Priority:
      1) CFB_TIX_REPO env
      2) %USERPROFILE%\\Documents\\football-ticket-tracker
      3) current working dir
      4) user profile
    """
    env_repo = os.environ.get("CFB_TIX_REPO")
    if env_repo and Path(env_repo).exists():
        return env_repo
    user = Path(os.environ.get("USERPROFILE", str(Path.home())))
    for cand in (
        user / "Documents" / "football-ticket-tracker",
        Path.cwd(),
        user,
    ):
        if cand.exists():
            return str(cand)
    return str(Path.cwd())


def _write_vbs(path: Path, target: str, args: str, shortcut_path: str, icon_path: str, working_dir: str = ""):
    vbs = f"""
    Set oWS = WScript.CreateObject("WScript.Shell")
    sLinkFile = "{shortcut_path}"
    Set oLink = oWS.CreateShortcut(sLinkFile)
    oLink.TargetPath = "{target}"
    oLink.Arguments  = "{args}"
    oLink.WorkingDirectory = "{working_dir}"
    oLink.IconLocation = "{icon_path}"
    oLink.WindowStyle = 7
    oLink.Description = "CFB Ticket Tracker"
    oLink.Save
    """
    path.write_text(textwrap.dedent(vbs).strip(), encoding="utf-8")


def create_shortcuts():
    # Executables created by [project.gui-scripts] in pyproject.toml
    scripts = _scripts_dir()
    gui_exe    = scripts / "Ticket-Price-Predictor.exe"
    daemon_exe = scripts / "CFB-Ticket-Tracker.exe"

    if not gui_exe.exists() or not daemon_exe.exists():
        raise RuntimeError(
            "GUI executables not found. Install first inside venv: pip install -e ."
        )

    # Prefer hyphen names (match your repo icons), tolerate underscores
    gui_icon = _pick_icon(["cfb-tix_gui.ico", "cfb_tix_gui.ico"])
    da_icon  = _pick_icon(["cfb-tix_daemon.ico", "cfb_tix_daemon.ico"])

    appdata = Path(os.environ.get("APPDATA", str(Path.home() / "AppData/Roaming")))
    start_menu_dir = appdata / r"Microsoft\Windows\Start Menu\Programs" / APP_DIR_NAME
    desktop_dir    = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
    startup_dir    = appdata / r"Microsoft\Windows\Start Menu\Programs\Startup"

    _ensure_dir(start_menu_dir)
    _ensure_dir(desktop_dir)
    _ensure_dir(startup_dir)

    start_gui_lnk    = start_menu_dir / "Ticket Price Predictor.lnk"
    start_daemon_lnk = start_menu_dir / "CFB Ticket Tracker (Background).lnk"
    desk_gui_lnk     = desktop_dir / "Ticket Price Predictor.lnk"
    boot_daemon_lnk  = startup_dir / "CFB-Ticket-Tracker.lnk"

    working_dir = _detect_working_dir()

    tmp = Path.cwd() / "_tmp_vbs"
    _ensure_dir(tmp)

    jobs = [
        (tmp / "gui_startmenu.vbs",    str(gui_exe),    "", str(start_gui_lnk),    gui_icon),
        (tmp / "daemon_startmenu.vbs", str(daemon_exe), "", str(start_daemon_lnk), da_icon),
        (tmp / "gui_desktop.vbs",      str(gui_exe),    "", str(desk_gui_lnk),     gui_icon),
        (tmp / "daemon_startup.vbs",   str(daemon_exe), "", str(boot_daemon_lnk),  da_icon),
    ]
    for vbs_path, target, args, lnk, icon in jobs:
        _write_vbs(vbs_path, target, args, lnk, icon, working_dir)

    # Create shortcuts silently (no PowerShell window)
    for vbs_path, *_ in jobs:
        os.system(f'cscript //nologo "{vbs_path}"')

    # Cleanup temp files (best-effort)
    try:
        for f in tmp.iterdir():
            f.unlink(missing_ok=True)
        tmp.rmdir()
    except Exception:
        pass


def main():
    try:
        create_shortcuts()
        print("✅ Shortcuts created: Start Menu, Desktop (GUI), and Startup (daemon).")
    except Exception as e:
        print(f"❌ Failed to create shortcuts: {e}")


if __name__ == "__main__":
    main()
