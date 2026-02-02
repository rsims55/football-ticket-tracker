# src/gui/ticket_predictor_gui.py — Month/Day x-axis; 4x daily bins; month labels; past-games show lowest price text
from __future__ import annotations

import sys
import os
from pathlib import Path
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QComboBox, QPushButton, QMessageBox, QListView, QSizePolicy, QScrollArea, QHBoxLayout
)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from matplotlib.ticker import Formatter, NullFormatter
from matplotlib.transforms import blended_transform_factory
import numpy as np

# -----------------------------
_THIS = Path(__file__).resolve()
PROJ_DIR = _THIS.parents[2]  # .../repo root
# Default to current year unless overridden via env var
SEASON_YEAR = int(os.getenv("SEASON_YEAR", str(datetime.now(ZoneInfo("America/New_York")).year if ZoneInfo else datetime.now().year)))

REPO_DATA_LOCK = os.getenv("REPO_DATA_LOCK", "1") == "1"
ALLOW_ESCAPE   = os.getenv("REPO_ALLOW_NON_REPO_OUT", "0") == "1"

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
    return PROJ_DIR / default_rel

SNAPSHOT_PATH = _resolve_file("SNAPSHOT_PATH", Path("data/daily") / f"price_snapshots_{SEASON_YEAR}.csv")
SNAPSHOT_FALLBACK = _resolve_file("SNAPSHOT_PATH_FALLBACK", Path("data/daily/price_snapshots.csv"))
PRED_PATH     = _resolve_file("OUTPUT_PATH",  Path("data/predicted/predicted_prices_optimal.csv"))
MERGED_PATH   = _resolve_file("MERGED_OUT",   Path("data/predicted/predicted_with_context.csv"))

# ---- Time-of-day bin labels (Night 00:00–05:59; Morning 06:00–11:59; Afternoon 12:00–17:59; Evening 18:00–23:59)
def tod_bucket(hour: int) -> str:
    if 6 <= hour <= 11:  return "Morning"
    if 12 <= hour <= 17: return "Afternoon"
    if 18 <= hour <= 23: return "Evening"
    return "Night"

class TODMinorFormatter(Formatter):
    def __call__(self, x, pos=None):
        dt = mdates.num2date(x)
        return tod_bucket(dt.hour)

class DayNoZeroFormatter(mdates.DateFormatter):
    """Formats day-of-month without leading zeros (portable)."""
    def __call__(self, x, pos=None):
        s = super().__call__(x, pos)  # usually '01'..'31'
        return s.lstrip('0')

def _fmt_mtime(p: Path) -> str:
    try:
        ts = p.stat().st_mtime
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except FileNotFoundError:
        return "missing"

def _season_state() -> str:
    now = datetime.now(ZoneInfo("America/New_York")) if ZoneInfo else datetime.now()
    year = now.year
    start = datetime(year, 8, 1)
    end = datetime(year + 1, 2, 1)
    return "In-season" if start <= now < end else "Offseason"


def get_status_text() -> str:
    season_state = _season_state()
    return (
        f"{season_state} • Snapshots: {_fmt_mtime(SNAPSHOT_PATH)}   |   "
        f"Predictions: {_fmt_mtime(PRED_PATH)}   |   Postseason excluded"
    )

# =========================================================

class TicketApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎟️ Ticket Price Predictor")
        self.setGeometry(200, 200, 1080, 780)
        self.setStyleSheet("QMainWindow { background-color: #f0f2f5; font-family: Arial; }")

        # state
        self.current_row = None
        self.current_event_id = None
        self.ax = None

        # trajectory state
        self.traj_times = []
        self.traj_prices = []
        self.playhead_line = None
        self.playhead_marker = None
        self.hover_annot = None
        self._tt_series = None
        self._yy_series = None

        try:
            self.snapshots = self.load_snapshot_data()
            self.df = self.load_and_merge_data()
            self.kickoff_lookup, self.schedule_df = self._build_kickoff_lookup()
        except Exception as e:
            QMessageBox.critical(self, "Startup Error", f"{e}")
            raise

        self.init_ui()

        # timers
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_countdown_live)
        self.timer.start(1000)

    # ---------------- Data ----------------
    def _infer_collected_dt(self, snaps: pd.DataFrame) -> pd.Series:
        for c in ["collected_at", "snapshot_datetime", "retrieved_at", "scraped_at", "collected_dt"]:
            if c in snaps.columns:
                ts = pd.to_datetime(snaps[c], errors="coerce")
                if ts.notna().any():
                    return ts
        time_cand = next((c for c in ["time_collected", "collection_time", "snapshot_time", "time_pulled"] if c in snaps.columns), None)
        if "date_collected" in snaps.columns and time_cand:
            ts = pd.to_datetime(snaps["date_collected"].astype(str).str.strip()+" "+snaps[time_cand].astype(str).str.strip(), errors="coerce")
            if ts.notna().any():
                return ts
        if "date_collected" in snaps.columns:
            ts = pd.to_datetime(snaps["date_collected"], errors="coerce")
            if ts.notna().any():
                return ts
        return pd.Series(pd.NaT, index=snaps.index)

    def _coerce_datetime(self, series: pd.Series) -> pd.Series:
        if series is None:
            return pd.Series(pd.NaT)
        s = series.copy()
        has_tz = s.astype(str).str.contains(r"(Z|[+-]\d{2}:?\d{2})").any()
        if has_tz:
            dt = pd.to_datetime(s, errors="coerce", utc=True)
        else:
            dt = pd.to_datetime(s, errors="coerce")
        if dt.dtype == object:
            dt = pd.to_datetime(s, errors="coerce", utc=True)
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_convert("America/New_York").dt.tz_localize(None)
        return dt

    def _is_postseason_row(self, df: pd.DataFrame) -> pd.Series:
        if "is_postseason" in df.columns:
            return df["is_postseason"].fillna(False).astype(bool)
        if "title" in df.columns:
            return df["title"].fillna("").astype(str).str.contains(
                r"\b(bowl|playoff|first round|quarterfinal|semifinal|final|championship|cfp)\b",
                case=False,
                regex=True,
            )
        return pd.Series(False, index=df.index)

    def _confidence_label(self, row: dict) -> str:
        # Simple heuristic based on feature completeness and time to game.
        try:
            hours = float(row.get("hours_until_game", np.nan))
        except Exception:
            hours = np.nan
        signals = [
            pd.notna(row.get("capacity")),
            pd.notna(row.get("week")),
            pd.notna(row.get("home_last_point_diff_at_snapshot")),
            pd.notna(row.get("away_last_point_diff_at_snapshot")),
        ]
        completeness = sum(1 for s in signals if s)
        if pd.notna(hours) and hours <= 168 and completeness >= 3:
            return "High"
        if pd.notna(hours) and hours <= 336 and completeness >= 2:
            return "Medium"
        return "Low"

    def _recommended_window(self, predicted_min_time: pd.Timestamp) -> str:
        if pd.isna(predicted_min_time):
            return "—"
        start = predicted_min_time - pd.Timedelta(hours=6)
        end = predicted_min_time + pd.Timedelta(hours=6)
        return f"{start.strftime('%a, %b %d %I:%M %p')} → {end.strftime('%a, %b %d %I:%M %p')}"

    def load_snapshot_data(self) -> pd.DataFrame:
        path = SNAPSHOT_PATH if SNAPSHOT_PATH.exists() else SNAPSHOT_FALLBACK
        if not path.exists():
            raise FileNotFoundError(f"Could not find snapshot CSV at '{SNAPSHOT_PATH}' or '{SNAPSHOT_FALLBACK}'")
        snaps = pd.read_csv(path, low_memory=False)
        snaps["lowest_price"] = pd.to_numeric(snaps.get("lowest_price"), errors="coerce")
        snaps["collected_dt"] = self._infer_collected_dt(snaps)
        if "startDateEastern" not in snaps.columns:
            if "date_local" in snaps.columns:
                if "time_local" in snaps.columns:
                    snaps["startDateEastern"] = pd.to_datetime(snaps["date_local"].astype(str)+" "+snaps["time_local"].astype(str), errors="coerce")
                else:
                    snaps["startDateEastern"] = pd.to_datetime(snaps["date_local"], errors="coerce")
        if "startDateEastern" in snaps.columns:
            snaps["startDateEastern"] = self._coerce_datetime(snaps["startDateEastern"])
        if "homeTeam" not in snaps.columns and "home_team_guess" in snaps.columns:
            snaps = snaps.rename(columns={"home_team_guess": "homeTeam"})
        if "awayTeam" not in snaps.columns and "away_team_guess" in snaps.columns:
            snaps = snaps.rename(columns={"away_team_guess": "awayTeam"})
        snaps = snaps[~self._is_postseason_row(snaps)].copy()
        # Filter to current season window (Aug 1 -> Feb 1 next year)
        if "startDateEastern" in snaps.columns:
            start = pd.Timestamp(f"{SEASON_YEAR}-08-01")
            end = pd.Timestamp(f"{SEASON_YEAR+1}-02-01")
            snaps = snaps[(snaps["startDateEastern"] >= start) & (snaps["startDateEastern"] < end)]
        return snaps

    def load_and_merge_data(self) -> pd.DataFrame:
        if MERGED_PATH.exists():
            merged = pd.read_csv(MERGED_PATH)
        else:
            if PRED_PATH.exists():
                pred = pd.read_csv(PRED_PATH)
            else:
                pred = None

            # Parse types from predictions
            if pred is not None and "startDateEastern" in pred.columns:
                pred["startDateEastern"] = self._coerce_datetime(pred["startDateEastern"])

            # New observed columns: ensure exist, parse
            if pred is not None:
                if "observed_lowest_price_num" not in pred.columns:
                    pred["observed_lowest_price_num"] = np.nan
                pred["observed_lowest_price_num"] = pd.to_numeric(pred["observed_lowest_price_num"], errors="coerce")

            if pred is not None and "observed_lowest_dt" not in pred.columns:
                pred["observed_lowest_dt"] = ""

            def _parse_iso(s):
                try:
                    return pd.to_datetime(s, errors="coerce")
                except Exception:
                    return pd.NaT

            if pred is not None:
                pred["observed_lowest_dt_parsed"] = pred["observed_lowest_dt"].apply(_parse_iso)

            if pred is not None and "event_id" not in pred.columns:
                raise KeyError("Predictions must contain 'event_id'.")

            # Merge snapshots (keeps your current behavior)
            snaps = getattr(self, "snapshots", None)
            if pred is not None and snaps is not None and "event_id" in snaps.columns:
                merged = pred.merge(snaps, on="event_id", how="left", suffixes=("", "_snap"))
                if "startDateEastern" not in merged and "startDateEastern_snap" in merged:
                    merged["startDateEastern"] = merged["startDateEastern_snap"]
            elif snaps is not None:
                merged = snaps.copy()
            else:
                merged = pred if pred is not None else pd.DataFrame()

        # Final cleanups
        if "startDateEastern" in merged.columns:
            merged["startDateEastern"] = self._coerce_datetime(merged["startDateEastern"])
        if "week" in merged.columns:
            merged["week"] = pd.to_numeric(merged["week"], errors="coerce").astype("Int64")
        for tcol in ("homeTeam", "awayTeam"):
            if tcol not in merged.columns:
                merged[tcol] = pd.NA
        if "predicted_lowest_price" not in merged.columns and "predicted_lowest_price_num" in merged.columns:
            merged["predicted_lowest_price"] = pd.to_numeric(merged["predicted_lowest_price_num"], errors="coerce")
        merged = merged[~self._is_postseason_row(merged)].copy()
        return merged

    def _build_kickoff_lookup(self):
        weekly_dir = PROJ_DIR / "data" / "weekly"
        files = sorted(weekly_dir.glob("full_*_schedule.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return {}, pd.DataFrame()
        f = files[0]
        try:
            cols = pd.read_csv(f, nrows=1).columns
            use_cols = [c for c in ["id", "homeTeam", "awayTeam", "startDateEastern", "kickoffTimeStr", "startDate", "game_date"] if c in cols]
            df = pd.read_csv(f, usecols=use_cols)
        except Exception:
            return {}, pd.DataFrame()
        if "id" not in df.columns:
            return {}, pd.DataFrame()
        df["id"] = df["id"].astype(str)
        lookup = {}
        for _, r in df.iterrows():
            lookup[str(r.get("id"))] = {
                "kickoffTimeStr": r.get("kickoffTimeStr"),
                "startDateEastern": r.get("startDateEastern"),
                "startDate": r.get("startDate"),
                "game_date": r.get("game_date"),
            }
        return lookup, df

    def _find_schedule_match(self, row):
        if self.schedule_df is None or self.schedule_df.empty:
            return None
        home = row.get("homeTeam")
        away = row.get("awayTeam")
        if not home or not away:
            return None
        df = self.schedule_df
        # Base matchup filter
        m = (df["homeTeam"] == home) & (df["awayTeam"] == away)
        if not m.any():
            # fallback: swapped home/away (neutral site or inconsistent sources)
            m = (df["homeTeam"] == away) & (df["awayTeam"] == home)
        cand = df[m]
        if cand.empty:
            return None

        # Try to match on date if available
        date_val = None
        for dcol in ("date_local", "startDateEastern", "startDate", "game_date"):
            if dcol in row and pd.notna(row[dcol]):
                date_val = pd.to_datetime(row[dcol], errors="coerce")
                if pd.notna(date_val):
                    date_val = date_val.date()
                    break
        if date_val is not None:
            def _row_date(x):
                dt = pd.to_datetime(x, errors="coerce")
                return dt.date() if pd.notna(dt) else None
            if "startDateEastern" in cand.columns:
                cand = cand.copy()
                cand["_date"] = cand["startDateEastern"].apply(_row_date)
                exact = cand[cand["_date"] == date_val]
                if not exact.empty:
                    return exact.iloc[0]
        return cand.iloc[0] if not cand.empty else None


    # ---------------- UI ----------------
    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)
        self.central_widget.setLayout(main_layout)

        self.setStyleSheet("""
            QMainWindow { background-color: #f3f5f7; font-family: "Segoe UI", Arial, sans-serif; }
            QLabel { color: #1f2933; }
            QComboBox { font-size: 14px; padding: 2px 8px; border: 1px solid #cbd2d9; border-radius: 6px; background: #ffffff; }
            QComboBox::drop-down { width: 26px; border: 0px; }
            QComboBox::down-arrow { width: 10px; height: 10px; }
            QPushButton { background-color: #1a73e8; color: white; padding: 10px 12px; font-size: 14px; font-weight: 700; border: none; border-radius: 8px; }
            QPushButton:hover { background-color: #1669c1; }
        """)

        # Top panel (controls + details)
        top_panel = QWidget()
        top_layout = QVBoxLayout()
        top_layout.setSpacing(10)
        top_panel.setLayout(top_layout)
        top_panel.setStyleSheet("""
            background-color: #ffffff;
            border-radius: 12px;
            padding: 16px;
            border: 1px solid #e4e7eb;
        """)
        main_layout.addWidget(top_panel, stretch=3)

        title = QLabel("🎟️ Ticket Price Predictor")
        title.setAlignment(Qt.AlignLeft)
        title.setStyleSheet("QLabel { font-size: 24px; font-weight: 800; margin-bottom: 4px; }")
        top_layout.addWidget(title)

        status = QLabel(get_status_text())
        status.setStyleSheet("QLabel { font-size: 12px; color: #52606d; }")
        top_layout.addWidget(status)

        # Controls row (compact)
        controls_row = QHBoxLayout()
        controls_row.setSpacing(12)

        self.home_combo = QComboBox(); self.home_combo.setView(QListView())
        self.home_combo.addItem("Select Home Team")
        homes = sorted(self.df["homeTeam"].dropna().unique()); self.home_combo.addItems(homes)
        self.home_combo.currentIndexChanged.connect(self.update_away_teams)
        
        self.home_combo.setMinimumHeight(50)
        self.home_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.away_combo = QComboBox(); self.away_combo.setView(QListView())
        self.away_combo.addItem("Select Away Team")
        self.away_combo.setMinimumHeight(50)
        self.away_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.predict_button = QPushButton()
        self.predict_button.setText("Run Prediction")
        self.predict_button.clicked.connect(self.get_prediction)
        self.predict_button.setFixedHeight(50)
        self.predict_button.setMinimumWidth(180)
        self.predict_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.predict_button.setStyleSheet(
            "QPushButton { background-color: #1a73e8; color: #ffffff; font-size: 14px; font-weight: 700; "
            "border: none; border-radius: 8px; padding: 8px 14px; }"
            "QPushButton:hover { background-color: #1669c1; }"
        )

        controls_row.addWidget(self.home_combo, 1)
        controls_row.addWidget(self.away_combo, 1)
        controls_row.addWidget(self.predict_button, 1)
        top_layout.addLayout(controls_row)

        # Details (scrollable; at least as tall as chart)
        self.details_label = QLabel("")
        self.details_label.setTextFormat(Qt.RichText)
        self.details_label.setWordWrap(True)
        self.details_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.details_label.setStyleSheet(
            "QLabel { background-color: #fafafa; border: 1px solid #e4e7eb; padding: 12px; font-size: 14px; border-radius: 8px; }"
        )
        self.details_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.details_scroll = QScrollArea()
        self.details_scroll.setWidget(self.details_label)
        self.details_scroll.setWidgetResizable(True)
        self.details_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.details_scroll.setStyleSheet("QScrollArea { border: 0; }")
        self.details_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        top_layout.addWidget(self.details_scroll, stretch=1)

        # Countdown
        self.countdown_label = QLabel(""); self.countdown_label.setTextFormat(Qt.RichText); self.countdown_label.setAlignment(Qt.AlignLeft)
        self.countdown_label.setStyleSheet("QLabel { font-size: 13px; color: #52606d; padding: 2px 0 8px 2px; }")
        top_layout.addWidget(self.countdown_label, stretch=0)

        # Bottom panel (chart)
        chart_panel = QWidget()
        chart_layout = QVBoxLayout()
        chart_layout.setSpacing(8)
        chart_panel.setLayout(chart_layout)
        chart_panel.setStyleSheet("""
            background-color: #ffffff;
            border-radius: 12px;
            padding: 12px;
            border: 1px solid #e4e7eb;
        """)
        main_layout.addWidget(chart_panel, stretch=2)

        # Chart
        self.chart_canvas = FigureCanvas(Figure(constrained_layout=True))
        self.chart_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.chart_canvas.setMinimumHeight(300)
        self.ax = self.chart_canvas.figure.add_subplot(111)
        chart_layout.addWidget(self.chart_canvas, stretch=1)

        # Hover
        self.chart_canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

    # ---------------- Dropdowns ----------------
    def update_away_teams(self):
        selected_home = self.home_combo.currentText()
        if selected_home.startswith("--"):
            self.away_combo.clear(); self.away_combo.addItem("-- Select Away Team --"); return
        df_home = self.df[self.df["homeTeam"] == selected_home].sort_values("startDateEastern")
        seen, ordered = set(), []
        for away in df_home["awayTeam"]:
            if pd.isna(away): continue
            if away not in seen: ordered.append(away); seen.add(away)
        self.away_combo.clear(); self.away_combo.addItem("-- Select Away Team --"); self.away_combo.addItems(ordered)

    # ---------------- Format helpers ----------------
    def _fmt_ampm_no_sec(self, val: str) -> str:
        if val is None or str(val).strip() == "" or str(val).strip().upper() == "TBD": return "TBD"
        s = str(val).strip()
        try:
            t = pd.to_datetime(s).time()
            return datetime.strptime(t.strftime("%H:%M"), "%H:%M").strftime("%I:%M %p").lstrip("0")
        except Exception:
            for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M%p"):
                try:
                    t = datetime.strptime(s, fmt).time()
                    return datetime.strptime(t.strftime("%H:%M"), "%H:%M").strftime("%I:%M %p").lstrip("0")
                except Exception:
                    continue
            return "TBD"

    def _fmt_full_date(self, dt_like) -> str:
        try:
            dt = pd.to_datetime(dt_like); return dt.strftime("%A, %B %d, %Y")
        except Exception:
            return "TBD"

    def _parse_time_hhmm(self, t_str: str):
        if not t_str: return None
        try:
            return pd.to_datetime(str(t_str)).time()
        except Exception:
            for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M%p"):
                try: return datetime.strptime(str(t_str), fmt).time()
                except Exception: continue
            return None

    def _humanize_delta(self, td: pd.Timedelta) -> str:
        total = int(td.total_seconds()); sign = "-" if total < 0 else ""; total = abs(total)
        d = total // 86400; h = (total % 86400) // 3600; m = (total % 3600) // 60; s = total % 60
        parts = []
        if d: parts.append(f"{d}d")
        if h or d: parts.append(f"{h}h")
        parts.append(f"{m}m"); parts.append(f"{s}s")
        return sign + " ".join(parts)

    def _to_eastern(self, ts_like):
        t = pd.to_datetime(ts_like, errors="coerce")
        if pd.isna(t):
            return t
        if ZoneInfo is None:
            return t
        try:
            if t.tzinfo is None:
                return t.tz_localize(ZoneInfo("America/New_York"))
            return t.tz_convert(ZoneInfo("America/New_York"))
        except Exception:
            return t

    def _fmt_time_from_dt(self, dt_like) -> str:
        dt = self._to_eastern(dt_like)
        try:
            if pd.isna(dt):
                return "TBD"
            t = pd.to_datetime(dt).time()
            return datetime.strptime(f"{t.hour:02d}:{t.minute:02d}", "%H:%M").strftime("%I:%M %p").lstrip("0")
        except Exception:
            return "TBD"

    def _format_kickoff(self, row) -> str:
        # Check schedule lookup by event_id/id first
        event_id = row.get("event_id") or row.get("id")
        if event_id is not None:
            ref = self.kickoff_lookup.get(str(event_id))
            if ref:
                for cand in ("kickoffTimeStr",):
                    v = ref.get(cand)
                    if pd.notna(v) and str(v).strip():
                        t = self._parse_time_hhmm(str(v))
                        if t:
                            return datetime.strptime(f"{t.hour:02d}:{t.minute:02d}", "%H:%M").strftime("%I:%M %p").lstrip("0")
                for dt_key in ("startDateEastern", "startDate", "game_date"):
                    v = ref.get(dt_key)
                    if pd.notna(v):
                        dt = self._to_eastern(v)
                        if pd.notna(dt):
                            t = pd.to_datetime(dt).time()
                            if t and not ((t.hour == 0 and t.minute == 0) or (t.hour == 3 and t.minute == 0)):
                                return datetime.strptime(f"{t.hour:02d}:{t.minute:02d}", "%H:%M").strftime("%I:%M %p").lstrip("0")

        # Fallback: match by teams/date in schedule file
        sched = self._find_schedule_match(row)
        if sched is not None:
            v = sched.get("kickoffTimeStr")
            if pd.notna(v) and str(v).strip():
                t = self._parse_time_hhmm(str(v))
                if t:
                    return datetime.strptime(f"{t.hour:02d}:{t.minute:02d}", "%H:%M").strftime("%I:%M %p").lstrip("0")
            for dt_key in ("startDateEastern", "startDate", "game_date"):
                if dt_key in sched and pd.notna(sched[dt_key]):
                    dt = self._to_eastern(sched[dt_key])
                    if pd.notna(dt):
                        t = pd.to_datetime(dt).time()
                        if t and not ((t.hour == 0 and t.minute == 0) or (t.hour == 3 and t.minute == 0)):
                            return datetime.strptime(f"{t.hour:02d}:{t.minute:02d}", "%H:%M").strftime("%I:%M %p").lstrip("0")

        # Prefer explicit time fields first (avoid placeholder midnight/3am)
        for cand in ("kickoffTimeStr", "kickoff_time", "time_local", "start_time", "startTime", "startTimeEastern"):
            if cand in row and pd.notna(row[cand]) and str(row[cand]).strip():
                t = self._parse_time_hhmm(str(row[cand]))
                if t:
                    return datetime.strptime(f"{t.hour:02d}:{t.minute:02d}", "%H:%M").strftime("%I:%M %p").lstrip("0")

        # Fall back to datetime fields
        base_dt = None
        for dt_key in ("startDateEastern", "start_date", "startDate", "date_local", "gameDate"):
            if dt_key in row and pd.notna(row[dt_key]):
                base_dt = self._to_eastern(row[dt_key])
                if pd.notna(base_dt):
                    break

        if base_dt is not None and pd.notna(base_dt):
            t = pd.to_datetime(base_dt).time()
            if t and not ((t.hour == 0 and t.minute == 0) or (t.hour == 3 and t.minute == 0)):
                return datetime.strptime(f"{t.hour:02d}:{t.minute:02d}", "%H:%M").strftime("%I:%M %p").lstrip("0")
        return "TBD"

    def _fmt_money(self, x) -> str:
        try:
            return f"${float(x):,.2f}"
        except Exception:
            return "—"

    def _fmt_dt(self, ts_like) -> str:
        try:
            ts = self._to_eastern(ts_like)
            if pd.isna(ts):
                return "—"
            return pd.to_datetime(ts).strftime("%A, %B %d, %Y %I:%M %p").lstrip("0").replace(" 0", " ")
        except Exception:
            return "—"

    # ---------------- Prediction & rendering ----------------
    def get_prediction(self):
        try:
            home = self.home_combo.currentText(); away = self.away_combo.currentText()
            if home.startswith("--") or away.startswith("--"):
                self.details_label.setText("<b>❌ Please select both a home and away team.</b>"); self.countdown_label.setText(""); return

            match = self.df[(self.df["homeTeam"] == home) & (self.df["awayTeam"] == away)].copy()
            match["startDateEastern"] = self._coerce_datetime(match["startDateEastern"])
            match = match.sort_values("startDateEastern")
            if match.empty:
                self.details_label.setText("<b>❌ No prediction found for this matchup.</b>"); self.countdown_label.setText(""); return

            now = pd.Timestamp.now(); upcoming = match[match["startDateEastern"] >= now]
            row = (upcoming.iloc[0] if not upcoming.empty else match.iloc[0]).to_dict()

            self.current_row = row; self.current_event_id = row.get("event_id")
            self.update_countdown_live()

            kickoff = pd.to_datetime(row.get("startDateEastern"))
            if pd.notna(kickoff) and kickoff < now:
                # Game is in the past → show text-only lowest price
                self.render_past_lowest(self.current_event_id, kickoff, row)
                return

            # Build trajectory via model (natural timestamps; no snapping)
            self.traj_times, self.traj_prices, warn = self.build_trajectory(row)
            if warn: self._set_details_with_warning(row, warn)
            else:    self.render_details(row)

            self.render_chart(self.current_event_id, reuse_axes=False)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not fetch prediction.\n{e}")

    def _set_details_with_warning(self, row, warn: str):
        base = self._details_html(row, forecast_min_text="—")
        self.details_label.setText(base + f"<div style='margin-top:8px;color:#b00020;'>⚠️ {warn}</div>")

    def render_details(self, row):
        # compute min from trajectory
        traj_min_txt = "—"
        predicted_min_price = np.nan
        predicted_min_time = pd.NaT
        if self.traj_times and self.traj_prices:
            idx = int(np.nanargmin(pd.to_numeric(pd.Series(self.traj_prices), errors="coerce")))
            ts_min = pd.to_datetime(self.traj_times[idx]); p_min = float(self.traj_prices[idx])
            traj_min_txt = f"${p_min:,.2f} on {ts_min.strftime('%a, %b %d, %Y %I:%M %p')}"
            predicted_min_price = p_min
            predicted_min_time = ts_min
        kickoff = pd.to_datetime(row.get("startDateEastern"), errors="coerce")
        if pd.notna(kickoff) and kickoff < pd.Timestamp.now():
            traj_min_txt = "Post-Kickoff"
            predicted_min_time = pd.NaT
            predicted_min_price = np.nan
        self.details_label.setText(self._details_html(row, traj_min_txt, predicted_min_price, predicted_min_time))

    def _details_html(self, row, forecast_min_text: str, predicted_min_price=np.nan, predicted_min_time=pd.NaT) -> str:
        game_dt = self._to_eastern(row["startDateEastern"])
        stadium = row.get("stadium", "Unknown Venue")
        kickoff = self._format_kickoff(row)
        opt_time_str = self._fmt_ampm_no_sec(row.get("optimal_purchase_time"))
        opt_date_iso = row.get("optimal_purchase_date", "")
        opt_date_fmt = self._fmt_full_date(opt_date_iso)
        week_str = int(row["week"]) if pd.notna(row.get("week")) else "—"

        # --- Predicted from model (CSV fields) ---
        predicted_price = predicted_min_price if pd.notna(predicted_min_price) else row.get("predicted_lowest_price", np.nan)

        # --- Observed (ever) from CSV or snapshots ---
        observed_price, obs_ts = self._observed_lowest(self.current_event_id, row=row, cutoff_dt=None)
        confidence = self._confidence_label(row)
        rec_window = "Post-Kickoff" if forecast_min_text == "Post-Kickoff" else self._recommended_window(predicted_min_time)

        # Banner if observed < predicted
        warn_html = ""
        try:
            if float(observed_price) < float(predicted_price):
                warn_html = (
                    "<div style='margin:0 0 8px 0; padding:6px 10px; "
                    "background:#fdecea; color:#b00020; border:1px solid #f5c6cb; "
                    "border-radius:6px; font-weight:700;'>"
                    "⚠️ Cheaper price already observed than the model’s predicted minimum."
                    "</div>"
                )
        except Exception:
            pass

        return f"""
            <div style="font-size: 15px; line-height: 1.6;">
                <h2 style="margin-bottom: 2px;">{row.get('homeTeam','?')} vs {row.get('awayTeam','?')}</h2>

                {warn_html}

                <div style="font-size: 16px; font-weight: 700; color: #6a1b9a; margin: 4px 0 8px 0;">
                    Current Forecasted Minimum: {forecast_min_text}
                </div> <br>

                <div style="margin-top:4px;">
                    <b>Recommended Buy Window:</b> {rec_window}<br>
                    <b>Confidence:</b> {confidence}
                </div>

                <div style="margin-top:10px; padding-top:10px; border-top:1px dashed #ddd;">
                    <b>Observed (ever):</b><br>
                    &nbsp;&nbsp;Price:&nbsp;{self._fmt_money(observed_price)}<br>
                    &nbsp;&nbsp;When:&nbsp;{self._fmt_dt(obs_ts)}<br>
                </div>

                <div style="margin-top:10px; padding-top:10px; border-top:1px dashed #ddd;">
                    <b>Game Week:</b> {week_str}<br>
                    <b>Game Date:</b> {game_dt.strftime('%A, %B %d, %Y')}<br>
                    <b>Kickoff Time:</b> {kickoff}<br>
                    <b>Venue:</b> {stadium}<br>
                </div>
            </div>
        """

    def build_trajectory(self, row: dict):
        from gui.predict_trajectory import predict_for_times
        warn = ""
        kickoff = pd.to_datetime(row.get("startDateEastern")); now = pd.Timestamp.now()
        if pd.isna(kickoff): return [], [], "Missing kickoff datetime for this event."
        start = max(now, now.floor('T')); end = kickoff
        if end <= start: start = end - pd.Timedelta(hours=6)
        # 6-hour grid is fine; times remain natural (no snapping to bins)
        times = pd.date_range(start=start, end=end, freq="6H")
        if len(times) < 2: times = pd.DatetimeIndex([start, end])
        try:
            prices = predict_for_times(row, list(times))
            return list(times), list(prices), warn
        except Exception as e:
            warn = f"Prediction failed: {e}"; return [], [], warn

    # ---------------- Chart + hover interactions ----------------
    def render_chart(self, event_id, reuse_axes: bool = True):
        import matplotlib.dates as mdates
        from matplotlib.transforms import blended_transform_factory

        if reuse_axes and self.ax is not None:
            self.ax.clear(); ax = self.ax
        else:
            self.chart_canvas.figure.clear()
            ax = self.chart_canvas.figure.add_subplot(111)
            self.ax = ax

        # Validate trajectory
        if not self.traj_times or not self.traj_prices or len(self.traj_times) != len(self.traj_prices):
            ax.set_title("No forecast trajectory available")
            ax.set_xlabel("Date"); ax.set_ylabel("Price ($)")
            self.chart_canvas.draw_idle(); return

        tt = pd.to_datetime(pd.Series(self.traj_times))
        yy = pd.to_numeric(pd.Series(self.traj_prices), errors="coerce")
        ok = tt.notna() & yy.notna()
        tt, yy = tt[ok], yy[ok]
        if tt.empty:
            ax.set_title("No forecast trajectory available")
            ax.set_xlabel("Date"); ax.set_ylabel("Price ($)")
            self.chart_canvas.draw_idle(); return

        # Plot
        ax.plot(tt, yy, linewidth=2.0, label="Predicted price trajectory")
        ax.scatter(tt, yy, s=18, alpha=0.9)

        # X-axis: force one tick per day with day number
        x_min = tt.min().normalize()
        x_max = (tt.max().normalize() + pd.Timedelta(days=1))
        ax.set_xlim(x_min, x_max)

        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
        ax.tick_params(axis="x", which="major", labelsize=11, pad=20, bottom=True, labelbottom=True)

        # Gridlines
        ax.grid(True, which="major", axis="both", alpha=0.28)

        ax.set_xlabel("Date"); ax.set_ylabel("Price ($)")
        ax.legend(loc="best")

        # Autoscale Y
        ax.relim(); ax.autoscale(axis='y', tight=False)

        # Month separators and labels
        trans = blended_transform_factory(ax.transData, ax.get_xaxis_transform())
        months = pd.date_range(x_min, x_max, freq="MS")
        for m in months:
            ax.axvline(m, color="gray", linestyle="--", alpha=0.65, linewidth=1.0, zorder=0)
            ax.text(
                m, -0.25, m.strftime("%b"),
                ha="center", va="top", fontsize=12, fontweight="bold", color="gray",
                transform=trans, clip_on=False,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.95),
                zorder=5
            )

        # Title/playhead
        x0, y0 = tt.iloc[0], float(yy.iloc[0])
        self.playhead_line = ax.axvline(x0, linestyle="--", alpha=0.6)
        self.playhead_marker = ax.plot([x0], [y0], marker="o", markersize=6)[0]

        # Hover annotation
        if self.hover_annot is None:
            self.hover_annot = ax.annotate(
                "", xy=(0,0), xytext=(12,12), textcoords="offset points",
                bbox=dict(boxstyle="round", fc="white", ec="#888"),
                arrowprops=dict(arrowstyle="->", color="#666")
            )
        self.hover_annot.set_visible(False)

        # Keep series for hover
        self._tt_series = tt.reset_index(drop=True)
        self._yy_series = yy.reset_index(drop=True)
        self._update_title_and_marker(0)

        # **Critical fix:** disable constrained layout and add padding
        self.chart_canvas.figure.set_constrained_layout(False)
        self.chart_canvas.figure.subplots_adjust(bottom=0.1)

        self.chart_canvas.draw_idle()


    def _update_title_and_marker(self, idx: int):
        idx = max(0, min(idx, len(self._tt_series)-1))
        ts = self._tt_series.iloc[idx]; pr = float(self._yy_series.iloc[idx])
        bucket = tod_bucket(ts.hour)
        title = f"Forecasted Price: ${pr:,.2f} — {ts.strftime('%a %b %d, %Y')} ({bucket})"
        self.ax.set_title(title)

        if self.playhead_line is not None: self.playhead_line.set_xdata([ts])
        if self.playhead_marker is not None:
            self.playhead_marker.set_xdata([ts]); self.playhead_marker.set_ydata([pr])

        self.ax.relim(); self.ax.autoscale(axis='y', tight=False)
        self.chart_canvas.draw_idle()

    def on_mouse_move(self, event):
        if event.inaxes != self.ax or self._tt_series is None:
            if self.hover_annot:
                self.hover_annot.set_visible(False); self.chart_canvas.draw_idle()
            return
        xdata = event.xdata
        if xdata is None: return
        xs = mdates.date2num(self._tt_series.dt.to_pydatetime())
        idx = int(np.argmin(np.abs(xs - xdata)))
        ts = self._tt_series.iloc[idx]; pr = float(self._yy_series.iloc[idx])

        # within ~4 hours of a point (since spacing is 6h)
        if abs(xs[idx] - xdata) < (1.0/6.0):
            self.hover_annot.xy = (ts, pr)
            self.hover_annot.set_text(f"${pr:,.2f}\n{tod_bucket(ts.hour)}")
            self.hover_annot.set_visible(True)
            self._update_title_and_marker(idx)
            self.chart_canvas.draw_idle()
        else:
            if self.hover_annot.get_visible():
                self.hover_annot.set_visible(False); self.chart_canvas.draw_idle()

    # ---------------- Past game text-only summary ----------------
    def render_past_lowest(self, event_id, kickoff: pd.Timestamp, row: dict):
        ax = self.ax or self.chart_canvas.figure.add_subplot(111)
        ax.clear(); ax.axis("off")

        observed_price, _obs_ts = self._observed_lowest(event_id, row=row, cutoff_dt=None)
        min_price_txt = self._fmt_money(observed_price)

        matchup = f"{row.get('homeTeam','?')} vs {row.get('awayTeam','?')}"
        date_str = pd.to_datetime(row.get("startDateEastern")).strftime("%A, %B %d, %Y") if pd.notna(row.get("startDateEastern")) else "Game date unknown"

        ax.text(0.5, 0.68, f"{matchup}", ha="center", va="center", fontsize=16, fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.54, f"Game Date: {date_str}", ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.text(0.5, 0.40, "Observed Lowest Price", ha="center", va="center", fontsize=13, color="#555", transform=ax.transAxes)
        ax.text(0.5, 0.30, min_price_txt, ha="center", va="center", fontsize=20, fontweight="bold", transform=ax.transAxes, color="#2e7d32")

        self.details_label.setText(self._details_html(row, forecast_min_text="—") + "<div style='margin-top:8px;color:#444;'>Game has concluded.</div>")
        self.chart_canvas.draw_idle()

    def _observed_from_snapshots(self, event_id, cutoff_dt=None):
        try:
            df = self.snapshots.copy()
            if "event_id" not in df.columns:
                return (np.nan, pd.NaT)
            df = df[df["event_id"] == event_id]
            if df.empty:
                return (np.nan, pd.NaT)
            df["collected_dt"] = pd.to_datetime(df.get("collected_dt"), errors="coerce")
            df["lowest_price"] = pd.to_numeric(df.get("lowest_price"), errors="coerce")
            if cutoff_dt is not None and df["collected_dt"].notna().any():
                df = df[df["collected_dt"] <= cutoff_dt]
            if df["lowest_price"].notna().any():
                idx = df["lowest_price"].idxmin()
                return (df.loc[idx, "lowest_price"], df.loc[idx, "collected_dt"])
        except Exception:
            return (np.nan, pd.NaT)
        return (np.nan, pd.NaT)

    def _observed_lowest(self, event_id, row=None, cutoff_dt=None):
        observed_price = np.nan
        obs_ts = pd.NaT
        if row is not None:
            observed_price = pd.to_numeric(row.get("observed_lowest_price_num", np.nan), errors="coerce")
            obs_ts = row.get("observed_lowest_dt_parsed", pd.NaT)
            if pd.isna(obs_ts) and "observed_lowest_dt" in row:
                obs_ts = pd.to_datetime(row.get("observed_lowest_dt"), errors="coerce")
        if (pd.isna(observed_price) or pd.isna(obs_ts)) and event_id:
            snap_price, snap_dt = self._observed_from_snapshots(event_id, cutoff_dt=cutoff_dt)
            if pd.isna(observed_price) and pd.notna(snap_price):
                observed_price = snap_price
            if pd.isna(obs_ts) and pd.notna(snap_dt):
                obs_ts = snap_dt
        if pd.isna(observed_price) and row is not None:
            for k in ("actual_lowest_price", "observed_lowest_price", "final_lowest_price"):
                if k in row and pd.notna(row[k]):
                    try:
                        observed_price = float(row[k])
                        break
                    except Exception:
                        continue
        return observed_price, obs_ts

    # ---------------- Live countdown ----------------
    def update_countdown_live(self):
        if not self.current_row: return
        opt_date_iso = self.current_row.get("optimal_purchase_date", "")
        parsed_time = self._parse_time_hhmm(self.current_row.get("optimal_purchase_time"))
        if not opt_date_iso or not parsed_time: self.countdown_label.setText(""); return
        opt_dt = pd.to_datetime(f"{opt_date_iso} {parsed_time.strftime('%H:%M:%S')}", errors="coerce")
        if pd.isna(opt_dt): self.countdown_label.setText(""); return
        now_dt = pd.Timestamp.now(); delta = opt_dt - now_dt; human = self._humanize_delta(delta)
        html = '<div style="font-weight:700; color:#c62828;">PAST OPTIMAL DATE</div>' if delta.total_seconds() < 0 else f'<div style="color:#2e7d32;">Countdown to Optimal Ticket Price: {human}</div>'
        self.countdown_label.setText(html)

# =========================================================

def main():
    app = QApplication(sys.argv)
    win = TicketApp(); win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
