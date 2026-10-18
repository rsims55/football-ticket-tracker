"""
Stage 2 of the two-stage pricing pipeline: online/incremental regression predicting
absolute price directly -- no fixed anchor, no monotonic constraint on
hours_until_game (see [[model]] in project memory for why the production CatBoost
model's timing predictions were structurally broken).

ARCHITECTURE (v2 -- bucketed, per user direction 2026-09-15): time-until-kickoff is
split into discrete windows (BUCKET_EDGES), and EACH window gets its own
independently-fit Recursive Least Squares model -- genuine interaction between
timing and every other feature (team, rivalry, rank, recent price, ...), not an
approximation via basis-function products. A v1 that used one shared model with a
spline-basis time encoding was tried first and rejected: because that model was
purely additive (log(price) = time_shape(t) + game_effects), every game's predicted
floor collapsed to the SAME instant relative to kickoff regardless of team/rivalry/
etc. -- only the price LEVEL varied by game, never the timing. Splitting into
buckets fixes this structurally: each bucket's coefficients are completely
independent, so a rivalry game's Wednesday-week-out price can move completely
differently from a non-rivalry game's, not just up or down together.

For a given game, ALL buckets' models are evaluated and whichever bucket predicts
the lowest price is the recommendation -- "multiple models for different time
points, determining the best fit per game" per that same direction. Each bucket
naturally supplies a TIGHT timing range (its own span, narrower near kickoff where
the interesting price action is) and a price range (computed directly from that
bucket's own RLS covariance matrix P, no separate uncertainty model needed).

Algorithm per bucket: Recursive Least Squares (RLS) with a forgetting factor, so
each bucket's model updates in-place on every new snapshot that falls in its window
instead of being fully retrained. Cold start uses one closed-form ridge-regression
solve per bucket over all historical data in that window (what RLS converges to
anyway, computed directly instead of looped row-by-row for speed);
online_price_update.py then applies true incremental RLS updates for everything
after that, routing each new row to its own bucket by its own hours_until_game.

Feature encoding (same schema in every bucket):
  - hours_until_game -> a plain linear term now (used to be a tent-basis "time
    feature" in v1; a bucket's own coefficients now capture timing effects, so this
    is just fine-grained position within the bucket's window).
  - homeTeam/awayTeam/homeConference/awayConference -> one-hot against a FIXED
    vocabulary (every team/conference seen in the training data), so the online
    model's dimensionality never has to grow mid-season.
  - Recent-price features (price_prev, price_roll3_mean, log1p'd; price_delta_pct)
    ARE included here -- legitimate for forecasting, unlike in
    price_driver_report.py's driver-attribution job. Missing values (an event's
    literal first-ever snapshot has no "previous price") get a 0 fill + a
    `_missing` indicator. IMPORTANT: any caller building a row for this model
    should go through load_enriched_snapshots()/robust_latest_row_per_event() below
    -- skipping that means EVERY prediction hits the missing flags and the (small
    but real) team/game signal gets swamped by that one artifact. Learned this
    wiring predict_price.py up: predictions had collapsed to ~the same price for
    every game until the loader was fixed.
  - Other numeric/boolean drivers validated in price_driver_report.py: capacity,
    week, kickoff_hour, kickoff_dayofweek, homeTeamRank, awayTeamRank, isRivalry,
    isRankedMatchup, neutralSite, home/away_last_point_diff_at_snapshot. Deliberately
    excludes the win/loss-at-snapshot counters and snapshots_so_far -- both were
    found collinear (VIF > 100) with week/hours_until_game in the Stage 1 work.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Time-until-kickoff bucket edges (hours before kickoff). Data-driven from the 2026
# season's own row counts (see [[pricing-redesign]]): narrower near kickoff, where
# both the interesting price action AND row density are highest.
#   0-12h: 1,079 | 12-24h: 1,070 | 24-48h: 2,200 | 48-96h: 4,307 | 96-168h: 7,113
#   168-336h: 17,067 | 336-672h: 39,938 | 672-1344h: 86,497 | 1344-2500h: 184,350
BUCKET_EDGES = [0, 12, 24, 48, 96, 168, 336, 672, 1344, 2500]
BUCKET_LABELS = [f"{BUCKET_EDGES[i]}-{BUCKET_EDGES[i+1]}h" for i in range(len(BUCKET_EDGES) - 1)]

# log1p(price) is bounded below by 0 (price can't be negative); clamp above too, at
# a value no real ticket will ever hit, purely so a numerically unstable bucket
# (see RLSState.update's covariance-windup note) can never produce `inf` downstream
# of expm1 -- caught in production when a corrupted P matrix did exactly that.
MAX_LOG_PRICE = float(np.log1p(50_000.0))


def _clamp_log_price(x: np.ndarray | float):
    return np.clip(x, 0.0, MAX_LOG_PRICE)

NUMERIC_FEATURES = [
    "hours_until_game", "capacity", "week", "kickoff_hour", "kickoff_dayofweek",
    "homeTeamRank", "awayTeamRank",
    "home_last_point_diff_at_snapshot", "away_last_point_diff_at_snapshot",
]
BOOLEAN_FEATURES = ["isRivalry", "isRankedMatchup", "neutralSite"]
# Recent-price features: log1p the raw-dollar ones so their scale matches the
# log1p(price) target; price_delta_pct is already scale-free.
PRICE_HISTORY_FEATURES = ["price_prev_log", "price_roll3_mean_log", "price_delta_pct"]
PRICE_HISTORY_MISSING_SOURCE = ["price_prev", "price_roll3_mean", "price_delta_pct"]
CATEGORICAL_FEATURES = ["homeTeam", "awayTeam", "homeConference", "awayConference"]

TARGET = "lowest_price"


def bucket_index_vec(hours) -> np.ndarray:
    """Vectorized: which bucket each hours-until-kickoff value falls into."""
    edges = np.asarray(BUCKET_EDGES, dtype=float)
    h = np.clip(np.asarray(hours, dtype=float), edges[0], edges[-1] - 1e-6)
    idx = np.searchsorted(edges, h, side="right") - 1
    return np.clip(idx, 0, len(edges) - 2)


def bucket_index(hours: float) -> int:
    return int(bucket_index_vec(np.array([hours]))[0])


@dataclass
class Vocab:
    teams: list[str]
    conferences: list[str]

    def to_dict(self) -> dict:
        return {"teams": self.teams, "conferences": self.conferences}

    @staticmethod
    def from_dict(d: dict) -> "Vocab":
        return Vocab(teams=list(d["teams"]), conferences=list(d["conferences"]))

    @staticmethod
    def build(df: pd.DataFrame) -> "Vocab":
        teams = sorted(set(df["homeTeam"].dropna().astype(str)) | set(df["awayTeam"].dropna().astype(str)))
        confs = sorted(set(df["homeConference"].dropna().astype(str)) | set(df["awayConference"].dropna().astype(str)))
        return Vocab(teams=teams, conferences=confs)


@dataclass
class RLSState:
    """One bucket's model: coefficients + covariance + residual-variance estimate
    (the last is what makes prediction intervals possible without a separate model)."""
    theta: np.ndarray                # (p,) coefficients
    P: np.ndarray                    # (p, p) covariance-like matrix
    sigma2: float = 1.0              # running estimate of residual variance, log-price space
    n_updates: int = 0

    def update(self, x: np.ndarray, y: float, lam: float = 0.999, p_diag_cap: float = 1e4,
               err_cap: float = 3.0) -> None:
        """One Recursive Least Squares step, in place.

        Stabilized against two related failure modes, both found in production
        (see [[pricing-redesign]]):
          1. "Covariance windup": dividing P by lam every step grows it by ~0.1%
             each update in any direction the data doesn't actively inform (sparser
             buckets, rarely-updated teams). After ~170K updates one bucket's P
             diagonal reached 3.7e21 and went NEGATIVE (impossible for a valid
             covariance matrix) -- overflowed expm1 downstream into literal `inf`.
             Fixed by re-symmetrizing (corrects float drift) and capping the
             diagonal (bounds the windup itself).
          2. Residual-driven divergence: one real but extreme price swing (a game
             that went $927 -> $55 between snapshots, |err|~2.8 in log space) was
             enough to nudge theta off its stable point -- and because an
             already-wrong theta mispredicts the NEXT row too, even normal
             incoming data then produces another large residual, correcting
             aggressively again. Self-reinforcing: one bucket's theta reached
             1.8e17 and sigma2 reached 1.8e30 this way, with P-capping alone doing
             nothing to stop it (theta's growth isn't a P problem). Clipping the
             residual itself -- standard "robust RLS" / bounded-influence update --
             stops both the original trigger and the spiral it causes: no single
             update, whatever its source, can move theta or sigma2 by more than a
             bounded step.
        """
        Px = self.P @ x
        denom = lam + x @ Px
        if not np.isfinite(denom) or abs(denom) < 1e-12:
            self.n_updates += 1   # count it as seen, but don't let a degenerate
            return                # denom (near-zero or non-finite) touch the state

        k = Px / denom
        err = float(np.clip(y - x @ self.theta, -err_cap, err_cap))
        theta_new = self.theta + k * err
        P_new = (self.P - np.outer(k, Px)) / lam
        P_new = (P_new + P_new.T) / 2.0
        diag = np.clip(np.diag(P_new), 1e-10, p_diag_cap)
        np.fill_diagonal(P_new, diag)
        sigma2_new = float(lam * self.sigma2 + (1 - lam) * (err ** 2))

        # Last line of defense: whatever the cause, never let a non-finite result
        # overwrite good state -- the two incidents this update already guards
        # against both started from a state that looked fine right up until it
        # didn't, so verifying the OUTPUT rather than trusting the inputs is the
        # only check that would have actually caught them before the fact.
        if not (np.all(np.isfinite(theta_new)) and np.all(np.isfinite(P_new)) and np.isfinite(sigma2_new)):
            self.n_updates += 1
            return

        self.theta, self.P, self.sigma2 = theta_new, P_new, sigma2_new
        self.n_updates += 1


@dataclass
class BucketedRLSState:
    buckets: dict[str, RLSState]     # bucket label -> its own independently-fit model
    feature_names: list[str]         # shared schema across all buckets
    vocab: Vocab
    watermark: str = ""              # ISO timestamp of the last snapshot processed
    n_updates: int = 0

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        sigma2s, bucket_updates = {}, {}
        for label, st in self.buckets.items():
            arrays[f"theta__{label}"] = st.theta
            arrays[f"P__{label}"] = st.P
            sigma2s[label] = float(st.sigma2)
            bucket_updates[label] = int(st.n_updates)
        np.savez(path, **arrays)
        meta = {
            "feature_names": self.feature_names,
            "vocab": self.vocab.to_dict(),
            "watermark": self.watermark,
            "n_updates": self.n_updates,
            "bucket_edges": BUCKET_EDGES,
            "bucket_labels": list(self.buckets.keys()),
            "sigma2": sigma2s,
            "bucket_n_updates": bucket_updates,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "BucketedRLSState":
        path = Path(path)
        npz = np.load(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        buckets = {}
        for label in meta["bucket_labels"]:
            buckets[label] = RLSState(
                theta=npz[f"theta__{label}"], P=npz[f"P__{label}"],
                sigma2=meta.get("sigma2", {}).get(label, 1.0),
                n_updates=meta.get("bucket_n_updates", {}).get(label, 0),
            )
        return BucketedRLSState(
            buckets=buckets,
            feature_names=meta["feature_names"],
            vocab=Vocab.from_dict(meta["vocab"]),
            watermark=meta.get("watermark", ""),
            n_updates=meta.get("n_updates", 0),
        )

    @staticmethod
    def exists(path: Path) -> bool:
        path = Path(path)
        return path.exists() and path.with_suffix(".json").exists()


def _feature_names(vocab: Vocab) -> list[str]:
    names = ["intercept"]
    names += NUMERIC_FEATURES
    names += [f"{c}_missing" for c in NUMERIC_FEATURES]
    names += BOOLEAN_FEATURES
    names += PRICE_HISTORY_FEATURES
    names += [f"{c}_missing" for c in PRICE_HISTORY_MISSING_SOURCE]
    names += [f"home__{t}" for t in vocab.teams]
    names += [f"away__{t}" for t in vocab.teams]
    names += [f"homeConf__{c}" for c in vocab.conferences]
    names += [f"awayConf__{c}" for c in vocab.conferences]
    return names


def _prep_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the log1p'd / missing-indicator columns this encoding needs."""
    df = df.copy()
    for c in ["price_prev", "price_roll3_mean"]:
        if c not in df.columns:
            df[c] = np.nan
        df[f"{c}_log"] = np.log1p(pd.to_numeric(df[c], errors="coerce").clip(lower=0))
    if "price_delta_pct" not in df.columns:
        df["price_delta_pct"] = np.nan
    for c in PRICE_HISTORY_MISSING_SOURCE:
        df[f"{c}_missing"] = pd.to_numeric(df[c], errors="coerce").isna().astype(float)
    for c in NUMERIC_FEATURES:
        if c not in df.columns:
            df[c] = np.nan
        df[f"{c}_missing"] = pd.to_numeric(df[c], errors="coerce").isna().astype(float)
    for c in BOOLEAN_FEATURES:
        if c not in df.columns:
            df[c] = False
        df[c] = df[c].fillna(False).infer_objects(copy=False).astype(float)
    return df


def encode_batch(df: pd.DataFrame, vocab: Vocab) -> np.ndarray:
    """Build the (n, p) design matrix for a batch of rows. Column order matches
    _feature_names(vocab) exactly -- every caller (training, backtest, live
    prediction) goes through this one function so they can never drift apart."""
    df = _prep_columns(df)
    n = len(df)
    blocks = [np.ones((n, 1))]                                   # intercept
    for c in NUMERIC_FEATURES:
        blocks.append(pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy().reshape(-1, 1))
    for c in NUMERIC_FEATURES:
        blocks.append(df[f"{c}_missing"].to_numpy().reshape(-1, 1))
    for c in BOOLEAN_FEATURES:
        blocks.append(df[c].to_numpy().reshape(-1, 1))
    for c in PRICE_HISTORY_FEATURES:
        blocks.append(pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy().reshape(-1, 1))
    for c in PRICE_HISTORY_MISSING_SOURCE:
        blocks.append(df[f"{c}_missing"].to_numpy().reshape(-1, 1))

    team_arr = np.array(vocab.teams)
    conf_arr = np.array(vocab.conferences)
    blocks.append((df["homeTeam"].astype(str).to_numpy()[:, None] == team_arr[None, :]).astype(float))
    blocks.append((df["awayTeam"].astype(str).to_numpy()[:, None] == team_arr[None, :]).astype(float))
    blocks.append((df["homeConference"].astype(str).to_numpy()[:, None] == conf_arr[None, :]).astype(float))
    blocks.append((df["awayConference"].astype(str).to_numpy()[:, None] == conf_arr[None, :]).astype(float))

    return np.hstack(blocks)


# --------------------------------------------------------------------- RLS math

def ridge_batch_fit_bucketed(df: pd.DataFrame, y_col: str, vocab: Vocab, ridge: float = 10.0,
                              chunk_size: int = 20_000) -> BucketedRLSState:
    """Closed-form ridge regression, ONE PER BUCKET, encoding and accumulating one
    chunk at a time so the full design matrix is never materialized at once.
    Residual variance (sigma2) per bucket is computed from accumulated sufficient
    statistics (y'y, X'y, X'X) rather than a second pass over the data.
    """
    feature_names = _feature_names(vocab)
    p = len(feature_names)
    XtX = {lbl: np.zeros((p, p)) for lbl in BUCKET_LABELS}
    Xty = {lbl: np.zeros(p) for lbl in BUCKET_LABELS}
    yty = {lbl: 0.0 for lbl in BUCKET_LABELS}
    counts = {lbl: 0 for lbl in BUCKET_LABELS}

    n = len(df)
    hours_all = pd.to_numeric(df["hours_until_game"], errors="coerce").fillna(0.0).to_numpy()
    bidx_all = bucket_index_vec(hours_all)

    for start in range(0, n, chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        Xc = encode_batch(chunk, vocab)
        yc = chunk[y_col].to_numpy()
        bidx = bidx_all[start:start + chunk_size]
        for i, lbl in enumerate(BUCKET_LABELS):
            mask = bidx == i
            if not mask.any():
                continue
            Xb, yb = Xc[mask], yc[mask]
            XtX[lbl] += Xb.T @ Xb
            Xty[lbl] += Xb.T @ yb
            yty[lbl] += float((yb ** 2).sum())
            counts[lbl] += int(mask.sum())

    buckets = {}
    for lbl in BUCKET_LABELS:
        A = XtX[lbl] + ridge * np.eye(p)
        P = np.linalg.inv(A)
        theta = P @ Xty[lbl]
        n_b = max(counts[lbl], 1)
        rss = yty[lbl] - 2 * theta @ Xty[lbl] + theta @ XtX[lbl] @ theta
        sigma2 = float(max(rss / max(n_b - 1, 1), 1e-6))
        buckets[lbl] = RLSState(theta=theta, P=P, sigma2=sigma2, n_updates=n_b)

    return BucketedRLSState(buckets=buckets, feature_names=feature_names, vocab=vocab)


def update_bucketed(state: BucketedRLSState, new_rows: pd.DataFrame, y_col: str = "_y",
                     lam: float = 0.999) -> BucketedRLSState:
    """Apply true incremental RLS updates, routing each new row to its own bucket's
    model by that row's own hours_until_game -- the whole point of the bucketed
    design: a rivalry game's near-kickoff price behavior updates that bucket's
    model without touching (or being diluted by) the far-out bucket's model."""
    new_rows = new_rows.sort_values("_snapshot_ts")
    hours = pd.to_numeric(new_rows["hours_until_game"], errors="coerce").fillna(0.0).to_numpy()
    bidx = bucket_index_vec(hours)
    X = encode_batch(new_rows, state.vocab)
    y = new_rows[y_col].to_numpy()

    for i in range(len(new_rows)):
        lbl = BUCKET_LABELS[bidx[i]]
        st = state.buckets[lbl]
        if X.shape[1] != len(st.theta):
            continue  # schema drift guard; shouldn't happen within a season
        st.update(X[i], float(y[i]), lam=lam)
    state.n_updates += len(new_rows)
    return state


# ---------------------------------------------------------------- prediction API

def _infer_kickoff(row: dict) -> pd.Timestamp:
    """startDateEastern is null on a real fraction of rows (~35% in the 2025+2026
    combined data) -- fall back through the same columns gui/predict_trajectory.py
    and predict_price.py do, or this silently drops events that resolve fine
    everywhere else in the codebase."""
    for key in ("startDateEastern", "kickoff_ts", "start_time", "startDate"):
        if key in row and pd.notna(row.get(key)):
            dt = pd.to_datetime(row[key], errors="coerce")
            if pd.notna(dt):
                return dt.tz_localize(None) if getattr(dt, "tzinfo", None) is not None else dt
    date_str = str(row.get("date_local", "")).strip()
    time_str = str(row.get("time_local", "")).strip()
    if date_str and date_str.lower() != "nan":
        dt_str = f"{date_str} {time_str}" if time_str and time_str.lower() != "nan" else date_str
        dt = pd.to_datetime(dt_str, errors="coerce")
        if pd.notna(dt):
            return dt.tz_localize(None) if getattr(dt, "tzinfo", None) is not None else dt
    return pd.NaT


def _with_kickoff_fields(row: dict, kickoff: pd.Timestamp) -> dict:
    r = dict(row)
    r["kickoff_hour"] = float(kickoff.hour)
    r["kickoff_dayofweek"] = float(kickoff.dayofweek)
    return r


def predict_price_range_online(row: dict, state: BucketedRLSState, z: float = 1.0) -> dict:
    """THE main per-game entry point. Evaluates every bucket's own model for this
    game and returns:
      - "all_buckets": a DataFrame, one row per bucket, with its predicted price and
        [price_low, price_high] interval (point +/- z * predictive std dev, computed
        directly from that bucket's own RLS P matrix and running sigma2 -- no
        separate uncertainty model needed).
      - "best": the bucket with the lowest predicted price, as a dict -- its own
        span becomes the recommended (tight, narrower near kickoff) BUY-BY-WHEN
        window, its interval becomes the recommended PRICE range.
    z=1.0 by default (~68% interval, "typical" band) -- override for a
    wider/narrower range; there's no single "right" width, so this is a knob, not a
    hardcoded constant.
    """
    kickoff = _infer_kickoff(row)
    if pd.isna(kickoff):
        return {}

    rows = []
    for i, lbl in enumerate(BUCKET_LABELS):
        lo, hi = BUCKET_EDGES[i], BUCKET_EDGES[i + 1]
        r = _with_kickoff_fields(row, kickoff)
        r["hours_until_game"] = (lo + hi) / 2.0
        x = encode_batch(pd.DataFrame([r]), state.vocab)[0]
        st = state.buckets[lbl]
        if x.shape[0] != len(st.theta):
            continue
        log_price = float(x @ st.theta)
        var = st.sigma2 * (1.0 + float(x @ st.P @ x))
        se = float(np.sqrt(var)) if np.isfinite(var) and var >= 0 else MAX_LOG_PRICE
        # The training target log1p(price) is bounded to [0, MAX_LOG_PRICE] (price
        # can't be negative, and no real ticket costs more than the generous ceiling
        # that bounds it at) -- clamp before expm1 rather than after, or a sparse/
        # noisy bucket (thin near-kickoff data especially, or a numerically unstable
        # P matrix -- see RLSState.update) can predict a negative price or overflow
        # expm1 into literal `inf`. Not a calibration fix, just enforcing physical
        # bounds at the one place every consumer reads from.
        rows.append({
            "bucket": lbl, "hours_lo": lo, "hours_hi": hi,
            "window_start": kickoff - pd.Timedelta(hours=hi),
            "window_end": kickoff - pd.Timedelta(hours=lo),
            "predicted_price": float(np.expm1(_clamp_log_price(log_price))),
            "price_low": float(np.expm1(_clamp_log_price(log_price - z * se))),
            "price_high": float(np.expm1(_clamp_log_price(log_price + z * se))),
            "n_training_rows": st.n_updates,
        })

    if not rows:
        return {}
    all_buckets = pd.DataFrame(rows)
    best = all_buckets.loc[all_buckets["predicted_price"].idxmin()]
    return {"all_buckets": all_buckets, "best": best.to_dict()}


def predict_price_online(row: dict, state: BucketedRLSState, times: Iterable[pd.Timestamp]) -> list[float]:
    """Point-price prediction at each requested time, mirroring
    gui/predict_trajectory.predict_for_times's signature so existing callers (chart
    plotting, etc.) work unchanged. Each queried time is routed to ITS OWN bucket's
    model -- predictions now genuinely vary by game AND by how far out you ask,
    not a shared curve every game converges to (see module docstring)."""
    times = list(times)
    kickoff = _infer_kickoff(row)
    if pd.isna(kickoff):
        return [np.nan] * len(times)

    hours_until = []
    for t in times:
        t = pd.Timestamp(t)
        if getattr(t, "tzinfo", None) is not None:
            t = t.tz_localize(None)
        hours_until.append(max(0.0, (kickoff - t).total_seconds() / 3600.0))

    batch_df = pd.DataFrame([_with_kickoff_fields(row, kickoff)] * len(times))
    batch_df["hours_until_game"] = hours_until
    X = encode_batch(batch_df, state.vocab)
    bidx = bucket_index_vec(np.array(hours_until))

    preds = np.full(len(times), np.nan)
    for i, lbl in enumerate(BUCKET_LABELS):
        mask = bidx == i
        if not mask.any():
            continue
        st = state.buckets[lbl]
        if X.shape[1] != len(st.theta):
            continue
        preds[mask] = X[mask] @ st.theta
    # Same clamp as predict_price_range_online: bound to [0, MAX_LOG_PRICE] before
    # expm1 so a raw linear prediction can't come out negative or overflow to inf.
    preds = np.where(np.isnan(preds), preds, _clamp_log_price(preds))
    return list(np.expm1(preds))


def predict_trajectory_online(row: dict, state: BucketedRLSState, hours_back: float = 2500,
                               step_hours: float = 6) -> tuple[pd.DataFrame, dict]:
    """Same shape as gui/predict_trajectory.predict_trajectory: scans a time grid
    back from kickoff and returns the predicted floor + when it occurs. Prefer
    predict_price_range_online for anything that wants the range/best-bucket info;
    this exists for callers (chart plotting) that just want a price curve."""
    kickoff = _infer_kickoff(row)
    if pd.isna(kickoff):
        raise ValueError("Missing/invalid kickoff timestamp.")

    hours = np.arange(1, int(hours_back) + 1, int(step_hours), dtype=int)
    ts_grid = [kickoff - pd.Timedelta(int(h), "h") for h in hours]
    prices = predict_price_online(row, state, ts_grid)

    df = pd.DataFrame({"when": ts_grid, "yhat": prices})
    summary = {
        "predicted_min_price": float(np.nanmin(df["yhat"])) if df["yhat"].notna().any() else np.nan,
        "predicted_min_time": df.loc[df["yhat"].idxmin(), "when"] if df["yhat"].notna().any() else pd.NaT,
    }
    return df, summary


# ------------------------------------------------------------------- shared data

def load_enriched_snapshots(restrict_to_season: int | None = None) -> pd.DataFrame:
    """Load all snapshots through train_catboost_min's feature pipeline so
    price_prev/price_roll3_mean/price_delta_pct/hours_until_game/etc. exist. ANY
    caller that's going to feed a row into this model -- training, backtesting, or a
    live prediction -- should build its anchor row from this, not a raw snapshot
    load (see module docstring for why).

    restrict_to_season, if given, limits to that YEAR's season window (Mar 1 -
    Feb 1), matching predict_price.py's convention -- otherwise this also pulls in
    other seasons in the combined snapshot files.
    """
    import modeling.train_catboost_min as tcm
    df = tcm._load_all_snapshots()
    df = df[~tcm._is_postseason_row(df)].copy()
    df = tcm._ensure_event_id(df)
    df = tcm._coerce_booleans(df)
    df = tcm._build_targets(df)
    # Drop post-kickoff scrapes before building price-history features -- some
    # seasons' data has them, and a post-game resale price isn't a real ticket price
    # (and, being chronologically latest, would otherwise get picked as the anchor).
    df = df[df["hours_until_game"] > 0].copy()
    if restrict_to_season is not None:
        y = restrict_to_season
        df = df[(df["_kickoff_ts"] >= f"{y}-03-01") & (df["_kickoff_ts"] < f"{y+1}-02-01")].copy()
    df = tcm._add_price_history_features(df)
    df["lowest_price"] = pd.to_numeric(df["lowest_price"], errors="coerce")
    df["collected_dt"] = df["_snapshot_ts"]   # alias so predict_price.py's helpers work unchanged
    return df


def robust_latest_row_per_event(df: pd.DataFrame, outlier_ratio: float = 4.0) -> pd.DataFrame:
    """Like predict_price._latest_row_per_event, but walks backward past a snapshot
    whose price is a wild outlier relative to THAT SAME EVENT's own trailing price
    history (price_roll3_mean) instead of blindly trusting "most recent" -- protects
    against a single bad scrape becoming the anchor without penalizing games that
    are consistently expensive (thin resale market), since those compare fine
    against their own history, not a global threshold."""
    d = df.sort_values(["event_id", "_snapshot_ts"], ascending=[True, False])
    price = pd.to_numeric(d["lowest_price"], errors="coerce")
    baseline = pd.to_numeric(d.get("price_roll3_mean"), errors="coerce")
    ratio = price / baseline
    sane = baseline.isna() | (baseline <= 0) | ((ratio >= 1.0 / outlier_ratio) & (ratio <= outlier_ratio))
    d = d.assign(_sane=sane & price.notna())

    picked = d[d["_sane"]].groupby("event_id", sort=False, as_index=False).head(1)
    missing_ids = set(d["event_id"]) - set(picked["event_id"])
    if missing_ids:
        # Every row for this event looked like an outlier relative to its own
        # history (e.g. only 1-2 snapshots ever taken) -- fall back to literal
        # most-recent rather than silently dropping the game.
        fallback = d[d["event_id"].isin(missing_ids)].groupby("event_id", sort=False, as_index=False).head(1)
        picked = pd.concat([picked, fallback], ignore_index=True)
    return picked.drop(columns=["_sane"])
