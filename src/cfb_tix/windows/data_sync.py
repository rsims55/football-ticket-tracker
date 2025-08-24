# src/cfb_tix/windows/data_sync.py
"""
All-in-one sync for price_snapshots.csv using a GitHub Release asset.

Features
- Pull (download) with ETag/Last-Modified caching
- Push (upload) to replace the asset on a release (creates release if missing)
- Optional interactive GitHub token prompt on Windows; saves to <repo>\\.env

Environment (override defaults as needed)
  SNAP_OWNER=rsims55
  SNAP_REPO=football-ticket-tracker
  SNAP_TAG=snapshots-latest
  SNAP_ASSET=price_snapshots.csv
  SNAP_DEST=<path to local csv>   (default: <repo>/data/daily/price_snapshots.csv)
  SNAP_GH_TOKEN or GITHUB_TOKEN   (required for upload)

CLI (examples)
  python -m cfb_tix.windows.data_sync ensure_token
  python -m cfb_tix.windows.data_sync pull
  python -m cfb_tix.windows.data_sync push
  python -m cfb_tix.windows.data_sync pull_push
"""

from __future__ import annotations

import os
import json
import shutil
import tempfile
import sys
from pathlib import Path
from typing import Optional, Tuple

import requests


# ----------------------- config -----------------------
OWNER  = os.getenv("SNAP_OWNER", "rsims55")
REPO   = os.getenv("SNAP_REPO", "football-ticket-tracker")
TAG    = os.getenv("SNAP_TAG", "snapshots-latest")
ASSET  = os.getenv("SNAP_ASSET", "price_snapshots.csv")
TOKEN  = os.getenv("SNAP_GH_TOKEN") or os.getenv("GITHUB_TOKEN")

# Resolve repo root robustly from this file location (…/src/cfb_tix/windows/data_sync.py)
# parents: [0]=windows, [1]=cfb_tix, [2]=src, [3]=<repo>
_THIS = Path(__file__).resolve()
_DEFAULT_ROOT = _THIS.parents[3]


def _find_repo_root() -> Path:
    # Look upward for pyproject.toml or .git or data/ directory; else fallback.
    p = _THIS
    for _ in range(6):
        p = p.parent
        if (p / "pyproject.toml").exists() or (p / ".git").exists() or (p / "data").exists():
            return p
    return _DEFAULT_ROOT


PROJ_DIR = _find_repo_root()
DEFAULT_DEST = PROJ_DIR / "data" / "daily" / "price_snapshots.csv"
DEST_PATH = Path(os.getenv("SNAP_DEST") or DEFAULT_DEST)
STATE_PATH = DEST_PATH.with_suffix(".etag.json")  # stores ETag/Last-Modified


# ----------------------- utils -----------------------
def _gh(headers: Optional[dict] = None) -> dict:
    h = {"Accept": "application/vnd.github+json"}
    tok = os.getenv("SNAP_GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    if headers:
        h.update(headers)
    return h


def _load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(etag: Optional[str], last_modified: Optional[str]):
    data = {"etag": etag, "last_modified": last_modified}
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _headers_for_conditional():
    st = _load_state()
    h = {}
    if st.get("etag"):
        h["If-None-Match"] = st["etag"]
    if st.get("last_modified"):
        h["If-Modified-Since"] = st["last_modified"]
    return h


def _atomic_write(src_tmp: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_final = dest.with_suffix(dest.suffix + ".tmp")
    shutil.move(str(src_tmp), str(tmp_final))
    os.replace(tmp_final, dest)  # atomic on Windows & POSIX


def _get_release_by_tag() -> Optional[dict]:
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/releases/tags/{TAG}"
    r = requests.get(url, headers=_gh(), timeout=30)
    if r.status_code == 200:
        return r.json()
    return None


def _create_release_if_missing() -> dict:
    rel = _get_release_by_tag()
    if rel:
        return rel
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/releases"
    payload = {
        "tag_name": TAG,
        "name": TAG,
        "body": "Automated snapshots release for price_snapshots.csv",
        "draft": False,
        "prerelease": False,
    }
    r = requests.post(url, headers=_gh({"Content-Type": "application/json"}), json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def _find_asset_in_release(rel: dict, name: str) -> Optional[dict]:
    for a in rel.get("assets", []):
        if a.get("name") == name:
            return a
    return None


def _delete_asset(asset_id: int) -> None:
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/releases/assets/{asset_id}"
    # 204 on success
    requests.delete(url, headers=_gh(), timeout=30)


# ----------------------- token prompt (Windows) -----------------------
def _read_dotenv(path: Path) -> dict:
    vals = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip()
    except Exception:
        pass
    return vals


def _write_dotenv(path: Path, key: str, value: str):
    vals = _read_dotenv(path)
    vals[key] = value
    lines = [f"{k}={v}" for k, v in vals.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_token_interactive(save_to_env: bool = True) -> Optional[str]:
    """
    On Windows, prompt for a GitHub token if none present in env or .env.
    Saves to <repo>\\.env as GITHUB_TOKEN=<token> (and sets SNAP_GH_TOKEN too).
    Returns the token or None if unavailable/cancelled.
    """
    # Check env
    tok = os.getenv("SNAP_GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    if tok:
        return tok

    # Check repo .env
    dotenv_path = PROJ_DIR / ".env"
    vals = _read_dotenv(dotenv_path)
    tok = vals.get("SNAP_GH_TOKEN") or vals.get("GITHUB_TOKEN")
    if tok:
        os.environ["GITHUB_TOKEN"] = tok
        os.environ["SNAP_GH_TOKEN"] = tok
        return tok

    # Try a minimal GUI prompt (tkinter) without console
    tok = None
    if os.name == "nt":
        try:
            import tkinter as tk
            from tkinter import simpledialog, messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo(
                "GitHub Token Required",
                "To upload snapshots to GitHub Releases, enter a Personal Access Token (repo scope).",
            )
            tok = simpledialog.askstring("GitHub Token", "Enter GitHub token (will be stored in .env):", show="*")
            root.destroy()
        except Exception:
            tok = None

    if tok and save_to_env:
        _write_dotenv(dotenv_path, "GITHUB_TOKEN", tok)
        os.environ["GITHUB_TOKEN"] = tok
        os.environ["SNAP_GH_TOKEN"] = tok
    return tok


# ----------------------- pull (download) -----------------------
def sync_down_latest_snapshots(verbose: bool = True) -> bool:
    """
    Returns True if a new file was downloaded, False if already up-to-date
    or if no remote asset exists.
    """
    rel = _get_release_by_tag()
    if not rel:
        if verbose:
            print(f"ℹ️  Release tag '{TAG}' not found in {OWNER}/{REPO}.")
        return False

    asset = _find_asset_in_release(rel, ASSET)
    if not asset or not asset.get("browser_download_url"):
        if verbose:
            print(f"ℹ️  Asset '{ASSET}' not found in release '{TAG}'.")
        return False

    dl_url = asset["browser_download_url"]

    # Try HEAD with conditionals
    req_headers = _gh(_headers_for_conditional())
    head = requests.head(dl_url, headers=req_headers, allow_redirects=True, timeout=30)
    if head.status_code == 304:
        if verbose:
            print("✅ Snapshots up-to-date (304 Not Modified).")
        return False

    # Fallback GET (some CDNs mishandle HEAD)
    if head.status_code >= 400:
        get_probe = requests.get(dl_url, headers=req_headers, stream=True, timeout=60)
        if get_probe.status_code == 304:
            if verbose:
                print("✅ Snapshots up-to-date (GET 304).")
            return False
        if get_probe.status_code >= 400:
            if verbose:
                print(f"⚠️  Download probe failed: {get_probe.status_code}")
            return False
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            for chunk in get_probe.iter_content(chunk_size=1 << 16):
                if chunk:
                    tf.write(chunk)
            tmp_path = Path(tf.name)
        _atomic_write(tmp_path, DEST_PATH)
        _save_state(get_probe.headers.get("ETag"), get_probe.headers.get("Last-Modified"))
        if verbose:
            print(f"⬇️  Downloaded latest → {DEST_PATH}")
        return True

    # HEAD OK -> do GET
    etag = head.headers.get("ETag")
    last_mod = head.headers.get("Last-Modified")
    r = requests.get(dl_url, headers=_gh(), stream=True, timeout=60)
    if r.status_code >= 400:
        if verbose:
            print(f"⚠️  Download failed: {r.status_code}")
        return False
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        for chunk in r.iter_content(chunk_size=1 << 16):
            if chunk:
                tf.write(chunk)
        tmp_path = Path(tf.name)
    _atomic_write(tmp_path, DEST_PATH)
    _save_state(etag, last_mod)
    if verbose:
        print(f"⬇️  Downloaded latest → {DEST_PATH}")
    return True


# ----------------------- push (upload) -----------------------
def upload_local_snapshots(verbose: bool = True) -> bool:
    """
    Upload local DEST_PATH to release TAG as ASSET (replace if exists).
    Returns True on success.
    """
    if not DEST_PATH.exists():
        if verbose:
            print(f"⚠️  Local snapshots missing: {DEST_PATH}")
        return False

    # Ensure we have a token; on Windows, prompt if missing.
    tok = os.getenv("SNAP_GH_TOKEN") or os.getenv("GITHUB_TOKEN") or ensure_token_interactive()
    if not tok:
        if verbose:
            print("⚠️  No GitHub token provided; cannot upload.")
        return False

    rel = _create_release_if_missing()
    upload_url_template = rel.get("upload_url", "")
    upload_url = upload_url_template.split("{", 1)[0] + f"?name={ASSET}"

    # Delete existing asset (best-effort)
    existing = _find_asset_in_release(rel, ASSET)
    if existing and existing.get("id"):
        _delete_asset(existing["id"])

    headers = _gh({"Content-Type": "text/csv"})
    with open(DEST_PATH, "rb") as f:
        r = requests.post(upload_url, headers=headers, data=f, timeout=120)

    if r.status_code in (200, 201):
        if verbose:
            size = r.json().get("size")
            print(f"⬆️  Uploaded {ASSET} ({size} bytes) to release '{TAG}'.")
        return True

    if verbose:
        print(f"⚠️  Upload failed: {r.status_code} {r.text[:200]}")
    return False


# ----------------------- combo -----------------------
def pull_then_push(verbose: bool = True) -> Tuple[bool, bool]:
    """
    Pull remote → upload local. Returns (pulled, pushed).
    """
    pulled = sync_down_latest_snapshots(verbose=verbose)
    pushed = upload_local_snapshots(verbose=verbose)
    return pulled, pushed


# ----------------------- CLI -----------------------
def _usage() -> int:
    print(
        "Usage: python -m cfb_tix.windows.data_sync [ensure_token|pull|push|pull_push]",
        file=sys.stderr,
    )
    return 2


def main():
    if len(sys.argv) < 2:
        sys.exit(_usage())
    cmd = sys.argv[1].lower()
    if cmd == "ensure_token":
        tok = ensure_token_interactive()
        print("✅ Token saved to .env" if tok else "⚠️ Token not provided.")
        return
    if cmd == "pull":
        sync_down_latest_snapshots(verbose=True)
        return
    if cmd == "push":
        upload_local_snapshots(verbose=True)
        return
    if cmd == "pull_push":
        pull_then_push(verbose=True)
        return
    sys.exit(_usage())


if __name__ == "__main__":
    main()
