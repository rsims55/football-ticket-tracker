#!/usr/bin/env python3
import os, sys, io, json, time, shutil
from datetime import datetime, timezone
from typing import List
import requests
import pandas as pd

# --- dotenv is optional; we try to load it if present ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:  # Python 3.11+
    import tomllib  # stdlib
except Exception:  # Python 3.10 and below
    import tomli as tomllib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CFG_PATH = os.path.join(ROOT, "config", "data_store.toml")

with open(CFG_PATH, "rb") as f:
    CFG = tomllib.load(f)

OWNER = CFG["store"]["owner"]
REPO  = CFG["store"]["repo"]
TAG   = CFG["store"]["release_tag"]
ASSET = CFG["store"]["asset_name"]

LOCAL_CSV   = os.path.join(ROOT, CFG["paths"]["local_csv"])
ARCHIVE_DIR = os.path.join(ROOT, CFG["paths"]["archive_dir"])

KEY_COLS: List[str] = CFG["merge"]["key_columns"]
TS_COL = CFG["merge"]["timestamp_col"]

SESSION = requests.Session()
TOKEN = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
if TOKEN:
    SESSION.headers["Authorization"] = f"Bearer {TOKEN}"
SESSION.headers["Accept"] = "application/vnd.github+json"
SESSION.headers["X-GitHub-Api-Version"] = "2022-11-28"

def _req(url, method="GET", **kw):
    r = SESSION.request(method, url, **kw)
    if r.status_code >= 400:
        raise RuntimeError(f"GitHub API error {r.status_code}: {r.text[:400]}")
    return r

def ensure_release():
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/releases/tags/{TAG}"
    r = SESSION.get(url)
    if r.status_code == 404:
        if not TOKEN:
            # no remote yet and can’t create → behave like “empty remote”
            return {"assets": []}
        payload = {"tag_name": TAG, "name": TAG, "prerelease": False}
        return _req(f"https://api.github.com/repos/{OWNER}/{REPO}/releases", "POST", json=payload).json()
    r.raise_for_status()
    return r.json()

def download_remote() -> pd.DataFrame:
    rel = ensure_release()
    asset = next((a for a in rel.get("assets", []) if a["name"] == ASSET), None)
    if not asset:
        return pd.DataFrame()
    r = SESSION.get(asset["browser_download_url"])
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))

def load_local() -> pd.DataFrame:
    if not os.path.exists(LOCAL_CSV):
        os.makedirs(os.path.dirname(LOCAL_CSV), exist_ok=True)
        return pd.DataFrame()
    return pd.read_csv(LOCAL_CSV)

def normalize_ts(df: pd.DataFrame) -> pd.DataFrame:
    if TS_COL in df.columns:
        df[TS_COL] = pd.to_datetime(df[TS_COL], errors="coerce", utc=True)
    return df

def merge(remote_df: pd.DataFrame, local_df: pd.DataFrame) -> pd.DataFrame:
    frames = [d for d in (remote_df, local_df) if not d.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if KEY_COLS and all(k in df.columns for k in KEY_COLS):
        if TS_COL in df.columns:
            df = normalize_ts(df).sort_values(TS_COL).drop_duplicates(KEY_COLS, keep="last")
        else:
            df = df.drop_duplicates(KEY_COLS, keep="last")
    else:
        df = df.drop_duplicates(keep="last")
    if TS_COL in df.columns:
        df = df.sort_values(TS_COL)
    return df

def write_local(df: pd.DataFrame):
    os.makedirs(os.path.dirname(LOCAL_CSV), exist_ok=True)
    if df.empty:
        df.to_csv(LOCAL_CSV, index=False)
        print(f"✅ Wrote (empty) {LOCAL_CSV}")
        return
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    df.to_csv(LOCAL_CSV, index=False)
    shutil.copy2(LOCAL_CSV, os.path.join(ARCHIVE_DIR, f"price_snapshots_{ts}.csv"))
    print(f"✅ Wrote {LOCAL_CSV} and archived copy.")

def upload(df: pd.DataFrame):
    if not TOKEN:
        print("ℹ️ No GH_TOKEN set; skipping upload.")
        return
    rel = ensure_release()
    upload_url = rel["upload_url"].split("{")[0]
    # delete old asset if present
    for a in rel.get("assets", []):
        if a["name"] == ASSET:
            _req(f"https://api.github.com/repos/{OWNER}/{REPO}/releases/assets/{a['id']}", "DELETE")
            break
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    r = SESSION.post(upload_url, params={"name": ASSET}, headers={"Content-Type": "text/csv"}, data=csv_bytes)
    if r.status_code >= 300:
        raise RuntimeError(f"Upload failed: {r.status_code} {r.text[:400]}")
    print(f"⬆️ Uploaded {ASSET} to release '{TAG}'.")

def main(mode="pull_push"):
    remote = download_remote()
    local = load_local()
    merged = merge(remote, local)
    write_local(merged)
    if mode in ("push", "pull_push"):
        upload(merged)

if __name__ == "__main__":
    # modes: pull (download only), push (upload only), pull_push (default)
    m = sys.argv[1] if len(sys.argv) > 1 else "pull_push"
    main(m)
