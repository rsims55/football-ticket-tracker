"""
Train a PyTorch sequence model (GRU) to predict:
  1) gap_pct: (lowest_price - future_min_price) / lowest_price
  2) time_to_min_hours (log1p)

The model consumes full per-event price trajectories and outputs a prediction
for every snapshot in the sequence. This captures temporal dynamics that
row-wise models miss.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
except Exception as e:  # pragma: no cover
    raise ImportError("PyTorch is required. Install with `pip install torch`.") from e


# -----------------------------
# Repo paths
# -----------------------------
def _find_repo_root(start: Path) -> Path:
    cur = start
    for p in [cur] + list(cur.parents):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return start.parent.parent


_THIS = Path(__file__).resolve()
PROJ_DIR = _find_repo_root(_THIS)

REPO_DATA_LOCK = os.getenv("REPO_DATA_LOCK", "1") == "1"
ALLOW_ESCAPE = os.getenv("REPO_ALLOW_NON_REPO_OUT", "0") == "1"


def _under_repo(p: Path) -> bool:
    try:
        return p.resolve().is_relative_to(PROJ_DIR.resolve())
    except AttributeError:
        return str(p.resolve()).startswith(str(PROJ_DIR.resolve()))


def _resolve_file(env_name: str, default_rel: Path) -> Path:
    env_val = os.getenv(env_name)
    if REPO_DATA_LOCK or not env_val:
        return PROJ_DIR / default_rel
    p = Path(env_val).expanduser()
    if _under_repo(p) or ALLOW_ESCAPE:
        return p
    print(f"🚫 {env_name} outside repo → {p} ; forcing repo path")
    return PROJ_DIR / default_rel


YEAR = int(os.getenv("SEASON_YEAR", "2025"))
TRAIN_ROWS_PATH = _resolve_file("TRAIN_ROWS_PATH", Path("data") / "modeling" / f"train_rows_{YEAR}.csv")
SNAPSHOT_PATH = _resolve_file("SNAPSHOT_PATH", Path("data") / "daily" / f"price_snapshots_{YEAR}.csv")
SNAPSHOT_FALLBACK = _resolve_file("SNAPSHOT_PATH_FALLBACK", Path("data") / "daily" / "price_snapshots.csv")
MODEL_PATH = _resolve_file("SEQ_MODEL_PATH", Path("models") / f"seq_gap_time_{YEAR}.pt")
REPORT_PATH = _resolve_file("SEQ_REPORT_PATH", Path("reports") / f"seq_model_report_{YEAR}.csv")


# -----------------------------
# Config
# -----------------------------
TEST_FRAC = float(os.getenv("SEQ_TEST_FRAC", "0.2"))
VAL_FRAC = float(os.getenv("SEQ_VAL_FRAC", "0.1"))
BATCH_SIZE = int(os.getenv("SEQ_BATCH_SIZE", "16"))
EPOCHS = int(os.getenv("SEQ_EPOCHS", "12"))
LR = float(os.getenv("SEQ_LR", "1e-3"))
HIDDEN_SIZE = int(os.getenv("SEQ_HIDDEN", "128"))
NUM_LAYERS = int(os.getenv("SEQ_LAYERS", "2"))
DROPOUT = float(os.getenv("SEQ_DROPOUT", "0.1"))
EARLY_STOP_PATIENCE = int(os.getenv("SEQ_EARLY_STOP_PATIENCE", "3"))
EARLY_STOP_MIN_DELTA = float(os.getenv("SEQ_EARLY_STOP_MIN_DELTA", "0.001"))

PRICE_GAP_CLIP_PCT = float(os.getenv("PRICE_GAP_CLIP_PCT", "99.5"))
PRICE_WEIGHT_MIN = float(os.getenv("PRICE_WEIGHT_MIN", "10"))
PRICE_WEIGHT_ALPHA = float(os.getenv("PRICE_WEIGHT_ALPHA", "1.5"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() and os.getenv("SEQ_USE_CUDA", "0") == "1" else "cpu")


# -----------------------------
# Feature schema
# -----------------------------
NUMERIC_FEATURES = [
    "hours_until_game",
    "days_until_game",
    "collection_hour_local",
    "lowest_price",
    "listing_count",
    "capacity",
    "neutralSite",
    "conferenceGame",
    "isRivalry",
    "isRankedMatchup",
    "homeTeamRank",
    "awayTeamRank",
    "week",
    "home_last_point_diff_at_snapshot",
    "away_last_point_diff_at_snapshot",
]

CATEGORICAL_FEATURES = ["homeTeam", "awayTeam"]


# -----------------------------
# Helpers — timestamps & coercions
# -----------------------------
def _best_snapshot_ts(df: pd.DataFrame) -> pd.Series:
    candidates = ["collected_at", "snapshot_datetime", "retrieved_at", "scraped_at"]
    time_only = ["time_collected", "collection_time", "snapshot_time"]
    ts = pd.Series(pd.NaT, index=df.index)
    for c in candidates:
        if c in df.columns:
            parsed = pd.to_datetime(df[c], errors="coerce")
            ts = ts.fillna(parsed)
    if "date_collected" in df.columns:
        date_raw = df["date_collected"].astype(str).str.strip()
        date_dt = pd.to_datetime(date_raw, errors="coerce", format="%m/%d/%Y")
        date_dt = date_dt.fillna(pd.to_datetime(date_raw, errors="coerce", format="%Y-%m-%d"))
        tcol = next((c for c in time_only if c in df.columns), None)
        if tcol:
            time_raw = df[tcol].astype(str).str.strip()
            time_norm = time_raw.str.replace(r"^(\d{1,2}:\d{2})$", r"\1:00", regex=True)
            time_td = pd.to_timedelta(time_norm, errors="coerce")
            ts = ts.fillna(date_dt.dt.normalize() + time_td)
        ts = ts.fillna(date_dt)
        # Last-chance parse for odd formats.
        ts = ts.fillna(pd.to_datetime(date_raw, errors="coerce"))
    # Fallback to game date/time if collection timestamps are missing.
    if ts.isna().any() and "date_local" in df.columns:
        dt_raw = df["date_local"].astype(str).str.strip()
        if "time_local" in df.columns:
            tm_raw = df["time_local"].astype(str).str.strip()
            dt_raw = dt_raw + " " + tm_raw
        ts = ts.fillna(pd.to_datetime(dt_raw, errors="coerce"))
    try:
        ts = ts.dt.tz_localize(None)
    except Exception:
        pass
    return ts


def _kickoff_ts(row: pd.Series) -> pd.Timestamp:
    date_str = str(row.get("date_local", "")).strip()
    time_str = str(row.get("time_local", "")).strip()
    dt_str = f"{date_str} {time_str}" if time_str and time_str.lower() != "nan" else date_str
    ts = pd.to_datetime(dt_str, errors="coerce")
    try:
        ts = ts.tz_localize(None)
    except Exception:
        pass
    return ts


def _coerce_booleans(df: pd.DataFrame) -> pd.DataFrame:
    truth_map = {
        True: True, False: False,
        "true": True, "false": False, "True": True, "False": False,
        "yes": True, "no": False, "YES": True, "NO": False,
        "y": True, "n": False, "Y": True, "N": False,
        1: True, 0: False, "1": True, "0": False,
        "t": True, "f": False, "T": True, "F": False,
        np.nan: np.nan,
    }
    for c in ["neutralSite", "conferenceGame", "isRivalry", "isRankedMatchup"]:
        if c in df.columns:
            s = df[c]
            if not pd.api.types.is_bool_dtype(s):
                s = s.map(truth_map).astype("boolean")
            df[c] = s.fillna(False).astype(bool)
    return df


def _ensure_event_id(df: pd.DataFrame) -> pd.DataFrame:
    if "event_id" not in df.columns:
        df["event_id"] = np.nan
    return df


# -----------------------------
# Targets
# -----------------------------
def _build_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_snapshot_ts"] = _best_snapshot_ts(df)
    df["_kickoff_ts"] = df.apply(_kickoff_ts, axis=1)
    df["hours_until_game"] = (df["_kickoff_ts"] - df["_snapshot_ts"]).dt.total_seconds() / 3600.0
    # If kickoff timestamps are missing, fall back to provided days_until_game.
    if "days_until_game" in df.columns:
        fallback_hours = pd.to_numeric(df["days_until_game"], errors="coerce") * 24.0
        df["hours_until_game"] = df["hours_until_game"].fillna(fallback_hours)
    df["days_until_game"] = df["hours_until_game"] / 24.0
    df["collection_hour_local"] = (
        df["_snapshot_ts"].dt.hour.fillna(0).astype(float)
        + df["_snapshot_ts"].dt.minute.fillna(0).astype(float) / 60.0
        + df["_snapshot_ts"].dt.second.fillna(0).astype(float) / 3600.0
    )

    df["lowest_price"] = pd.to_numeric(df["lowest_price"], errors="coerce")
    df = df[df["event_id"].notna()].copy()
    df = df.sort_values(["event_id", "_snapshot_ts"])

    future_min_price = []
    time_to_min = []

    for _, g in df.groupby("event_id", sort=False):
        prices = g["lowest_price"].to_numpy()
        times = g["_snapshot_ts"].to_numpy()

        prices_for_min = np.where(np.isnan(prices), np.inf, prices)
        suffix_min = np.minimum.accumulate(prices_for_min[::-1])[::-1]
        suffix_min = np.where(np.isinf(suffix_min), np.nan, suffix_min)

        min_idx = []
        current_min = np.inf
        current_idx = None
        for i in range(len(prices) - 1, -1, -1):
            p = prices_for_min[i]
            if np.isinf(p):
                min_idx.append(current_idx)
                continue
            if p <= current_min:
                current_min = p
                current_idx = i
            min_idx.append(current_idx)
        min_idx = list(reversed(min_idx))

        future_min_price.extend(suffix_min.tolist())
        for i, idx in enumerate(min_idx):
            if idx is None or pd.isna(times[i]) or pd.isna(times[idx]):
                time_to_min.append(np.nan)
            else:
                dt = (times[idx] - times[i]) / np.timedelta64(1, "h")
                time_to_min.append(float(dt))

    df["future_min_price"] = future_min_price
    df["time_to_min_hours"] = time_to_min

    # Completed games: use event min as ground truth
    now_ts = pd.Timestamp.now().tz_localize(None)
    kickoff = df.groupby("event_id")["_kickoff_ts"].min()
    complete = kickoff < now_ts
    df["event_complete"] = df["event_id"].map(complete).fillna(False).astype(bool)

    event_min_price = df.groupby("event_id")["lowest_price"].min()
    df["event_min_price"] = df["event_id"].map(event_min_price)
    try:
        event_min_ts = df.groupby("event_id", group_keys=False).apply(
            lambda g: g.loc[g["lowest_price"] == g["lowest_price"].min(), "_snapshot_ts"].min(),
            include_groups=False,
        )
    except TypeError:
        event_min_ts = df.groupby("event_id", group_keys=False).apply(
            lambda g: g.loc[g["lowest_price"] == g["lowest_price"].min(), "_snapshot_ts"].min()
        )

    if df["event_complete"].any():
        mask = df["event_complete"] & df["event_min_price"].notna()
        df.loc[mask, "future_min_price"] = df.loc[mask, "event_min_price"]
        dt = (df.loc[mask, "event_id"].map(event_min_ts).values - df.loc[mask, "_snapshot_ts"]) / np.timedelta64(1, "h")
        df.loc[mask, "time_to_min_hours"] = np.maximum(dt, 0)

    gap_abs = (df["lowest_price"] - df["future_min_price"]).clip(lower=0)
    gap_pct = gap_abs / df["lowest_price"].replace(0, np.nan)
    df["gap_pct"] = gap_pct.clip(lower=0, upper=1)
    if df["gap_pct"].notna().any():
        clip_val = df["gap_pct"].quantile(PRICE_GAP_CLIP_PCT / 100.0)
        df["gap_pct"] = df["gap_pct"].clip(upper=clip_val)

    df["time_to_min_hours_log"] = np.log1p(df["time_to_min_hours"].clip(lower=0))
    return df


# -----------------------------
# Dataset
# -----------------------------
@dataclass
class EventSeq:
    event_id: str
    X_num: np.ndarray
    home_id: int
    away_id: int
    y_gap: np.ndarray
    y_time: np.ndarray
    mask_gap: np.ndarray
    mask_time: np.ndarray
    lowest_price: np.ndarray


class SeqDataset(Dataset):
    def __init__(self, events: List[EventSeq]):
        self.events = events

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int) -> EventSeq:
        return self.events[idx]


def _collate(batch: List[EventSeq]):
    max_len = max(e.X_num.shape[0] for e in batch)
    num_feat = batch[0].X_num.shape[1]

    X = torch.zeros(len(batch), max_len, num_feat)
    y_gap = torch.zeros(len(batch), max_len)
    y_time = torch.zeros(len(batch), max_len)
    mask_gap = torch.zeros(len(batch), max_len)
    mask_time = torch.zeros(len(batch), max_len)
    low_price = torch.zeros(len(batch), max_len)
    home_ids = torch.tensor([e.home_id for e in batch], dtype=torch.long)
    away_ids = torch.tensor([e.away_id for e in batch], dtype=torch.long)

    for i, e in enumerate(batch):
        L = e.X_num.shape[0]
        X[i, :L, :] = torch.from_numpy(e.X_num)
        y_gap[i, :L] = torch.from_numpy(e.y_gap)
        y_time[i, :L] = torch.from_numpy(e.y_time)
        mask_gap[i, :L] = torch.from_numpy(e.mask_gap)
        mask_time[i, :L] = torch.from_numpy(e.mask_time)
        low_price[i, :L] = torch.from_numpy(e.lowest_price)

    return X, home_ids, away_ids, y_gap, y_time, mask_gap, mask_time, low_price


# -----------------------------
# Model
# -----------------------------
class SeqModel(nn.Module):
    def __init__(self, num_features: int, n_home: int, n_away: int, emb_dim: int = 16):
        super().__init__()
        self.home_emb = nn.Embedding(n_home, emb_dim)
        self.away_emb = nn.Embedding(n_away, emb_dim)
        self.gru = nn.GRU(
            input_size=num_features + emb_dim * 2,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
            batch_first=True,
        )
        self.head_gap = nn.Sequential(nn.Linear(HIDDEN_SIZE, 1))
        self.head_time = nn.Sequential(nn.Linear(HIDDEN_SIZE, 1))

    def forward(self, x_num, home_id, away_id):
        emb = torch.cat([self.home_emb(home_id), self.away_emb(away_id)], dim=-1)
        emb_exp = emb.unsqueeze(1).expand(-1, x_num.shape[1], -1)
        x = torch.cat([x_num, emb_exp], dim=-1)
        out, _ = self.gru(x)
        gap = torch.sigmoid(self.head_gap(out)).squeeze(-1)
        time = torch.nn.functional.softplus(self.head_time(out)).squeeze(-1)
        return gap, time


# -----------------------------
# Metrics
# -----------------------------
def _bucketed_price_mae(y_true: np.ndarray, y_pred: np.ndarray, bins=None) -> dict:
    if bins is None:
        bins = [0, 20, 50, 100, 200, np.inf]
    labels = [f"${bins[i]}–{bins[i+1]}" for i in range(len(bins) - 1)]
    out = {}
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_true >= lo) & (y_true < hi)
        if mask.any():
            out[f"price_mae_{labels[i]}"] = float(np.nanmean(np.abs(y_true[mask] - y_pred[mask])))
        else:
            out[f"price_mae_{labels[i]}"] = np.nan
    return out


# -----------------------------
# Build sequences
# -----------------------------
def _build_events(df: pd.DataFrame) -> Tuple[List[EventSeq], Dict[str, int], Dict[str, int], pd.DataFrame]:
    df = _ensure_event_id(df)
    df = _coerce_booleans(df)
    if "gap_pct" not in df.columns or "time_to_min_hours_log" not in df.columns:
        df = _build_targets(df)
    else:
        # Ensure expected fields exist when using prebuilt dataset.
        if "_snapshot_ts" not in df.columns:
            df["_snapshot_ts"] = _best_snapshot_ts(df)
        if "_kickoff_ts" not in df.columns:
            df["_kickoff_ts"] = df.apply(_kickoff_ts, axis=1)
        if "hours_until_game" not in df.columns:
            df["hours_until_game"] = (df["_kickoff_ts"] - df["_snapshot_ts"]).dt.total_seconds() / 3600.0
        if "days_until_game" not in df.columns:
            df["days_until_game"] = df["hours_until_game"] / 24.0

    # Filter usable rows
    df = df[df["hours_until_game"].notna()].copy()
    df = df[df["lowest_price"].notna()].copy()
    df = df[df["_snapshot_ts"].notna()].copy()

    # Categorical vocab
    home_vocab = {"<UNK>": 0}
    away_vocab = {"<UNK>": 0}
    for name, vocab in [("homeTeam", home_vocab), ("awayTeam", away_vocab)]:
        if name in df.columns:
            for v in df[name].dropna().astype(str).unique().tolist():
                if v not in vocab:
                    vocab[v] = len(vocab)

    # Missing indicators for numeric features
    for c in NUMERIC_FEATURES:
        if c in df.columns:
            df[f"{c}_missing"] = df[c].isna().astype(int)

    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns] + [f"{c}_missing" for c in NUMERIC_FEATURES if c in df.columns]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df[num_cols] = df[num_cols].fillna(0.0)

    events = []
    for event_id, g in df.groupby("event_id", sort=False):
        g = g.sort_values("_snapshot_ts")
        if len(g) < 2:
            continue
        X_num = g[num_cols].to_numpy(dtype=np.float32)
        home_id = home_vocab.get(str(g["homeTeam"].iloc[0]), 0)
        away_id = away_vocab.get(str(g["awayTeam"].iloc[0]), 0)
        y_gap = g["gap_pct"].to_numpy(dtype=np.float32)
        y_time = g["time_to_min_hours_log"].to_numpy(dtype=np.float32)
        mask_gap = np.isfinite(y_gap).astype(np.float32)
        mask_time = np.isfinite(y_time).astype(np.float32)
        low_price = g["lowest_price"].to_numpy(dtype=np.float32)
        events.append(EventSeq(event_id, X_num, home_id, away_id, y_gap, y_time, mask_gap, mask_time, low_price))

    return events, home_vocab, away_vocab, df


def _split_events(events: List[EventSeq], df: pd.DataFrame) -> Tuple[List[EventSeq], List[EventSeq], List[EventSeq]]:
    kickoff = df.groupby("event_id")["_kickoff_ts"].min().dropna().sort_values()
    ordered = kickoff.index.tolist()
    n_test = max(1, int(len(ordered) * TEST_FRAC))
    test_ids = set(ordered[-n_test:])
    train_ids = set(ordered[:-n_test])

    train_events = [e for e in events if e.event_id in train_ids]
    test_events = [e for e in events if e.event_id in test_ids]
    # If an event has no kickoff time, keep it in training by default.
    for e in events:
        if e.event_id not in train_ids and e.event_id not in test_ids:
            train_events.append(e)

    # Validation split from train
    n_val = max(1, int(len(train_events) * VAL_FRAC))
    val_events = train_events[-n_val:]
    train_events = train_events[:-n_val]
    return train_events, val_events, test_events


# -----------------------------
# Train
# -----------------------------
def train():
    if TRAIN_ROWS_PATH.exists():
        df = pd.read_csv(TRAIN_ROWS_PATH)
        print(f"[seq_model] Using prebuilt train rows: {TRAIN_ROWS_PATH}")
    elif SNAPSHOT_PATH.exists():
        df = pd.read_csv(SNAPSHOT_PATH)
    elif SNAPSHOT_FALLBACK.exists():
        df = pd.read_csv(SNAPSHOT_FALLBACK)
    else:
        raise FileNotFoundError("No snapshot file found.")

    events, home_vocab, away_vocab, df_used = _build_events(df)
    train_events, val_events, test_events = _split_events(events, df_used)

    def _count_rows(evts: List[EventSeq]) -> int:
        return int(sum(e.X_num.shape[0] for e in evts))

    def _count_mask(evts: List[EventSeq], which: str) -> int:
        if which == "gap":
            return int(sum(e.mask_gap.sum() for e in evts))
        return int(sum(e.mask_time.sum() for e in evts))

    total_rows = _count_rows(events)
    train_rows = _count_rows(train_events)
    val_rows = _count_rows(val_events)
    test_rows = _count_rows(test_events)

    gap_tr = _count_mask(train_events, "gap")
    gap_va = _count_mask(val_events, "gap")
    gap_te = _count_mask(test_events, "gap")
    time_tr = _count_mask(train_events, "time")
    time_va = _count_mask(val_events, "time")
    time_te = _count_mask(test_events, "time")

    def _pct(x: int, denom: int) -> str:
        return f"{(x / denom * 100):.1f}%" if denom > 0 else "NA"

    print(
        f"[seq_model] rows: total={total_rows} train={train_rows} val={val_rows} test={test_rows} | "
        f"gap_valid train/val/test={gap_tr}/{gap_va}/{gap_te} ({_pct(gap_tr,train_rows)}/{_pct(gap_va,val_rows)}/{_pct(gap_te,test_rows)}) | "
        f"time_valid train/val/test={time_tr}/{time_va}/{time_te} ({_pct(time_tr,train_rows)}/{_pct(time_va,val_rows)}/{_pct(time_te,test_rows)})"
    )

    train_loader = DataLoader(SeqDataset(train_events), batch_size=BATCH_SIZE, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(SeqDataset(val_events), batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate)
    test_loader = DataLoader(SeqDataset(test_events), batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate)

    num_feat = train_events[0].X_num.shape[1]
    model = SeqModel(num_feat, len(home_vocab), len(away_vocab)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.SmoothL1Loss(reduction="none")

    def _run_epoch(loader, train_mode=True):
        model.train(train_mode)
        total = 0.0
        count = 0.0
        for X, home_id, away_id, y_gap, y_time, m_gap, m_time, low_price in loader:
            X = X.to(DEVICE)
            home_id = home_id.to(DEVICE)
            away_id = away_id.to(DEVICE)
            y_gap = y_gap.to(DEVICE)
            y_time = y_time.to(DEVICE)
            m_gap = m_gap.to(DEVICE)
            m_time = m_time.to(DEVICE)
            low_price = low_price.to(DEVICE)

            if train_mode:
                opt.zero_grad()

            pred_gap, pred_time = model(X, home_id, away_id)

            # Weight gap loss to emphasize low-price games
            price_scale = (1.0 / torch.clamp(low_price, min=PRICE_WEIGHT_MIN)) ** PRICE_WEIGHT_ALPHA
            gap_loss = loss_fn(pred_gap, y_gap) * m_gap * price_scale
            time_loss = loss_fn(pred_time, y_time) * m_time
            loss = (gap_loss.sum() + time_loss.sum()) / torch.clamp(m_gap.sum() + m_time.sum(), min=1.0)

            if train_mode:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            total += loss.item()
            count += 1.0
        return total / max(count, 1.0)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        tr = _run_epoch(train_loader, True)
        va = _run_epoch(val_loader, False)
        print(f"epoch {epoch:02d} | train_loss={tr:.4f} | val_loss={va:.4f}")
        if va + EARLY_STOP_MIN_DELTA < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= EARLY_STOP_PATIENCE:
                print(f"[seq_model] Early stopping at epoch {epoch} (best_val={best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    # Evaluation
    model.eval()
    all_true_min = []
    all_pred_min = []
    all_time_true = []
    all_time_pred = []
    with torch.no_grad():
        for X, home_id, away_id, y_gap, y_time, m_gap, m_time, low_price in test_loader:
            X = X.to(DEVICE)
            home_id = home_id.to(DEVICE)
            away_id = away_id.to(DEVICE)
            pred_gap, pred_time = model(X, home_id, away_id)
            pred_gap = pred_gap.cpu().numpy()
            pred_time = pred_time.cpu().numpy()

            y_gap = y_gap.numpy()
            y_time = y_time.numpy()
            m_gap = m_gap.numpy().astype(bool)
            m_time = m_time.numpy().astype(bool)
            low_price = low_price.numpy()

            true_min = low_price * (1.0 - y_gap)
            pred_min = low_price * (1.0 - pred_gap)

            all_true_min.append(true_min[m_gap])
            all_pred_min.append(pred_min[m_gap])
            all_time_true.append(np.expm1(y_time[m_time]))
            all_time_pred.append(np.expm1(pred_time[m_time]))

    true_min = np.concatenate(all_true_min) if all_true_min else np.array([])
    pred_min = np.concatenate(all_pred_min) if all_pred_min else np.array([])
    t_true = np.concatenate(all_time_true) if all_time_true else np.array([])
    t_pred = np.concatenate(all_time_pred) if all_time_pred else np.array([])

    price_mae = float(np.nanmean(np.abs(true_min - pred_min))) if len(true_min) else np.nan
    price_rmse = float(np.sqrt(np.nanmean((true_min - pred_min) ** 2))) if len(true_min) else np.nan
    price_within = float(np.nanmean(np.abs(true_min - pred_min) / np.maximum(1e-6, np.abs(true_min)) <= 0.05)) if len(true_min) else np.nan

    time_mae = float(np.nanmean(np.abs(t_true - t_pred))) if len(t_true) else np.nan
    time_within = float(np.nanmean(np.abs(t_true - t_pred) <= 24.0)) if len(t_true) else np.nan

    bucket = _bucketed_price_mae(true_min, pred_min)

    report = {
        "year": YEAR,
        "rows_total_used": int(len(true_min)),
        "rows_total_all": total_rows,
        "rows_train": train_rows,
        "rows_val": val_rows,
        "rows_test": test_rows,
        "price_mae": price_mae,
        "price_rmse": price_rmse,
        "price_within_5pct": price_within,
        "time_mae_hours": time_mae,
        "time_within_24h": time_within,
        **bucket,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([report]).to_csv(REPORT_PATH, index=False)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_features": num_feat,
            "home_vocab": home_vocab,
            "away_vocab": away_vocab,
            "numeric_features": NUMERIC_FEATURES,
        },
        MODEL_PATH,
    )

    print("\nSTATUS: SUCCESS | PyTorch sequence model trained")
    print("-" * 72)
    for k, v in report.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print("-" * 72)


if __name__ == "__main__":
    train()
