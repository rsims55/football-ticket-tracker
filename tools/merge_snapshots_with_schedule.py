#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, re, sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import pandas as pd

# ---------- paths ----------
THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
DEFAULT_SNAP = ROOT / "data" / "daily" / "price_snapshots.csv"
DEFAULT_SCHEDULE = ROOT / "data" / "weekly" / "full_2025_schedule.csv"
DEFAULT_ALIASES = ROOT / "data" / "permanent" / "team_aliases.json"

# schedule → snapshot columns we’ll ATTACH (as *_sched). Snapshot columns are never overwritten.
CARRY_COLS = [
    "startDateEastern","week","stadium","capacity","neutralSite","conferenceGame",
    "isRivalry","isRankedMatchup","homeTeamRank","awayTeamRank",
    "homeConference","awayConference","homeTeam","awayTeam",
]

_STOP = {"university","state","college","the","of","and","at","football","st","saint"}
_VS_RE = re.compile(r"\s+vs\.?\s+", re.IGNORECASE)

def autodf(path: Path) -> pd.DataFrame:
    for kw in (dict(sep="\t", engine="python"), dict(engine="python"), {}):
        try: return pd.read_csv(path, **kw)
        except Exception: pass
    raise

def write_atomic(df: pd.DataFrame, path: Path, backup: bool=False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup and path.exists():
        path.with_suffix(path.suffix + ".bak").write_bytes(path.read_bytes())
    tmp = path.with_suffix(path.suffix + ".__tmp__")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

def load_aliases(p: Path) -> Dict[str,str]:
    try:
        raw = json.loads(Path(p).read_text(encoding="utf-8"))
        return {str(k).strip().lower(): str(v).strip() for k,v in raw.items()}
    except Exception:
        return {}

def canonicalize(name: Optional[str], aliases: Dict[str,str]) -> Optional[str]:
    if not isinstance(name, str): return name
    return aliases.get(name.strip().lower(), name.strip())

def normalize_team_name(s: Optional[str]) -> str:
    if not isinstance(s, str): return ""
    x = s.lower()
    for ch in "/.,-()[]{}'’":
        x = x.replace(ch, " ")
    x = re.sub(r"\s+", " ", x).strip()
    toks = [t for t in x.split(" ") if t and t not in _STOP]
    return " ".join(toks)

def tokens(s: Optional[str], aliases: Dict[str,str]) -> Set[str]:
    if not isinstance(s, str): return set()
    can = canonicalize(s, aliases)
    norm = normalize_team_name(can)
    return set(norm.split()) if norm else set()

def split_title_vs(title: Optional[str]) -> Tuple[Optional[str],Optional[str]]:
    if not isinstance(title, str) or not title.strip(): return (None, None)
    t = title.strip().replace("\u00A0", " ")
    if ":" in t: t = t.split(":", 1)[-1].strip() or t
    parts = _VS_RE.split(t, maxsplit=1)
    return (parts[0].strip(" -–—"), parts[1].strip(" -–—")) if len(parts)==2 else (None, None)

def pick_column(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    for n in names:
        if n in df.columns: return n
    return None

def bool_str(v) -> Optional[str]:
    if pd.isna(v): return None
    return "TRUE" if bool(v) else "FALSE"

# ---------- schedule prep ----------
REQ_SCHEDULE = {
    "startDateEastern": ["startDateEastern","start_date_eastern","startDate","game_date","date_local"],
    "homeTeam": ["homeTeam","home_team","home"],
    "awayTeam": ["awayTeam","away_team","away"],
}
OPT_SCHEDULE = {
    "week": ["week","game_week"],
    "stadium": ["stadium","venue","venue_name"],
    "capacity": ["capacity"],
    "neutralSite": ["neutralSite","neutral_site","neutral"],
    "conferenceGame": ["conferenceGame","conference_game","conference"],
    "isRivalry": ["isRivalry","rivalry"],
    "isRankedMatchup": ["isRankedMatchup","ranked_matchup"],
    "homeTeamRank": ["homeTeamRank","home_rank"],
    "awayTeamRank": ["awayTeamRank","away_rank"],
    "homeConference": ["homeConference","home_conference","home_conf"],
    "awayConference": ["awayConference","away_conference","away_conf"],
}

def prepare_schedule(df: pd.DataFrame, aliases: Dict[str,str]) -> pd.DataFrame:
    ren = {}
    for canon, cands in REQ_SCHEDULE.items():
        src = pick_column(df, cands)
        if not src:
            raise ValueError(f"Schedule missing required column for {canon}")
        if src != canon: ren[src] = canon
    for canon, cands in OPT_SCHEDULE.items():
        src = pick_column(df, cands)
        if src and src != canon: ren[src] = canon
    if ren: df = df.rename(columns=ren)

    for canon in OPT_SCHEDULE:
        if canon not in df.columns: df[canon] = pd.NA

    df["startDateEastern"] = pd.to_datetime(df["startDateEastern"], errors="coerce")
    df["date_key"] = df["startDateEastern"].dt.strftime("%Y-%m-%d")
    df["home_tok"] = df["homeTeam"].map(lambda s: tokens(s, aliases))
    df["away_tok"] = df["awayTeam"].map(lambda s: tokens(s, aliases))
    return df

# ---------- snapshot prep ----------
def first_existing(df: pd.DataFrame, *names: str) -> Optional[str]:
    for n in names:
        if n in df.columns: return n
    return None

def prepare_snapshots(df: pd.DataFrame, aliases: Dict[str,str]) -> pd.DataFrame:
    d = df.copy()
    if "homeTeam" not in d.columns: d["homeTeam"] = pd.NA
    if "awayTeam" not in d.columns: d["awayTeam"] = pd.NA

    if "title" in d.columns:
        th = d["title"].map(lambda t: split_title_vs(t)[0])
        ta = d["title"].map(lambda t: split_title_vs(t)[1])
        d["homeTeam"] = d["homeTeam"].fillna(th)
        d["awayTeam"] = d["awayTeam"].fillna(ta)

    d["homeTeam"] = d["homeTeam"].map(lambda s: canonicalize(s, aliases) if isinstance(s,str) else s)
    d["awayTeam"] = d["awayTeam"].map(lambda s: canonicalize(s, aliases) if isinstance(s,str) else s)

    date_col = first_existing(d, "date_local","date","game_date")
    if not date_col: raise ValueError("Snapshots missing a date column (expected 'date_local').")
    d["date_key"] = pd.to_datetime(d[date_col], errors="coerce").dt.strftime("%Y-%m-%d")

    d["home_tok"] = d["homeTeam"].map(lambda s: tokens(s, aliases))
    d["away_tok"] = d["awayTeam"].map(lambda s: tokens(s, aliases))
    return d

# ---------- matching ----------
def jacc(a: Set[str], b: Set[str]) -> float:
    if not a and not b: return 0.0
    return len(a & b) / max(1, len(a | b))

def score_pair(snap_row: pd.Series, sched_row: pd.Series) -> Tuple[float, bool]:
    sh, sa = snap_row["home_tok"], snap_row["away_tok"]
    if not sh or not sa:
        th, ta = split_title_vs(snap_row.get("title"))
        sh = sh or set(th.split()) if th else set()
        sa = sa or set(ta.split()) if ta else set()
    ch, ca = sched_row["home_tok"], sched_row["away_tok"]

    direct = (jacc(sh, ch) + jacc(sa, ca)) / 2.0
    flip = (jacc(sh, ca) + jacc(sa, ch)) / 2.0
    flipped = flip > direct
    return max(direct, flip), flipped

# ---------- merge (attach-only), write back to snapshots ----------
def attach_and_overwrite_snapshots(
    snap_df: pd.DataFrame,
    sched_df: pd.DataFrame,
    aliases: Dict[str,str],
    suffix: str = "_sched",
    min_match: float = 0.45,
    verbose: bool = False,
) -> pd.DataFrame:
    sched = prepare_schedule(sched_df, aliases)
    snap = prepare_snapshots(snap_df, aliases)

    sched_by_date = {dk: g.copy() for dk,g in sched.groupby("date_key")}
    out = snap_df.copy()  # keep original schema/values

    # ensure suffixed columns exist (they’ll be (re)created on overwrite)
    for col in CARRY_COLS + ["kickoff_time_eastern_sched"]:
        name = f"{col}{suffix}"
        if name not in out.columns:
            out[name] = pd.NA

    for i in list(snap.index):
        r = snap.loc[i]
        dk = r.get("date_key")
        if not isinstance(dk, str) or not dk: continue
        cands = sched_by_date.get(dk)
        if cands is None or cands.empty: continue

        scored = [(score_pair(r, c), c) for _, c in cands.iterrows()]
        scored.sort(key=lambda x: x[0][0], reverse=True)
        if not scored: continue

        (best_score, _flipped), best = scored[0]
        if best_score < min_match:
            if verbose: print(f"[skip] weak match score={best_score:.2f} row={i}")
            continue

        for col in CARRY_COLS:
            val = best.get(col)
            if col in ("neutralSite","conferenceGame","isRivalry","isRankedMatchup"):
                val = bool_str(val)
            out.at[i, f"{col}{suffix}"] = val

        sde = best.get("startDateEastern")
        if pd.notna(sde):
            try:
                t = pd.to_datetime(sde)
                out.at[i, "kickoff_time_eastern_sched"] = t.strftime("%H:%M")
            except Exception:
                pass

    return out

def main():
    ap = argparse.ArgumentParser(description="Attach schedule fields to snapshots (in-place). Schedule is never modified.")
    ap.add_argument("--snap", type=Path, default=DEFAULT_SNAP, help="Snapshots CSV to update (will be overwritten).")
    ap.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    ap.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    ap.add_argument("--suffix", type=str, default="_sched")
    ap.add_argument("--min-match", type=float, default=0.45)
    ap.add_argument("--backup", action="store_true", help="Write a .bak beside the snapshots before overwriting.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not args.snap.exists(): print(f"❌ snapshots not found: {args.snap}"); sys.exit(2)
    if not args.schedule.exists(): print(f"❌ schedule not found: {args.schedule}"); sys.exit(2)

    aliases = load_aliases(args.aliases)
    snap_df = autodf(args.snap)
    sched_df = autodf(args.schedule)

    updated = attach_and_overwrite_snapshots(
        snap_df=snap_df,
        sched_df=sched_df,
        aliases=aliases,
        suffix=args.suffix,
        min_match=args.min_match,
        verbose=args.verbose,
    )
    write_atomic(updated, args.snap, backup=args.backup)
    print(f"✅ updated snapshots in-place: {args.snap} (rows: {len(updated):,})")

if __name__ == "__main__":
    main()
