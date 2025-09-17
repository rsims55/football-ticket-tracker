# src/gui/ticket_predictor_gui.py — Month/Day x-axis; 4x daily bins; month labels; past-games show lowest price text
from __future__ import annotations

import sys
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QComboBox, QPushButton, QMessageBox, QListView, QSizePolicy, QScrollArea
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

SNAPSHOT_PATH = _resolve_file("SNAPSHOT_PATH", Path("data/daily/price_snapshots.csv"))
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

def get_status_text() -> str:
    return f"Snapshots: {_fmt_mtime(SNAPSHOT_PATH)}   |   Predictions: {_fmt_mtime(PRED_PATH)}"

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

    def load_snapshot_data(self) -> pd.DataFrame:
        if not SNAPSHOT_PATH.exists():
            raise FileNotFoundError(f"Could not find snapshot CSV at '{SNAPSHOT_PATH}'")
        snaps = pd.read_csv(SNAPSHOT_PATH)
        snaps["lowest_price"] = pd.to_numeric(snaps.get("lowest_price"), errors="coerce")
        snaps["collected_dt"] = self._infer_collected_dt(snaps)
        if "startDateEastern" not in snaps.columns:
            if "date_local" in snaps.columns:
                if "time_local" in snaps.columns:
                    snaps["startDateEastern"] = pd.to_datetime(snaps["date_local"].astype(str)+" "+snaps["time_local"].astype(str), errors="coerce")
                else:
                    snaps["startDateEastern"] = pd.to_datetime(snaps["date_local"], errors="coerce")
        if "homeTeam" not in snaps.columns and "home_team_guess" in snaps.columns:
            snaps = snaps.rename(columns={"home_team_guess": "homeTeam"})
        if "awayTeam" not in snaps.columns and "away_team_guess" in snaps.columns:
            snaps = snaps.rename(columns={"away_team_guess": "awayTeam"})
        return snaps

    def load_and_merge_data(self) -> pd.DataFrame:
        if MERGED_PATH.exists():
            merged = pd.read_csv(MERGED_PATH)
        else:
            if not PRED_PATH.exists():
                raise FileNotFoundError(f"Could not find predictions CSV at '{PRED_PATH}'")
            pred = pd.read_csv(PRED_PATH)

            # Parse types from predictions
            if "startDateEastern" in pred.columns:
                pred["startDateEastern"] = pd.to_datetime(pred["startDateEastern"], errors="coerce")

            # New observed columns: ensure exist, parse
            if "observed_lowest_price_num" not in pred.columns:
                pred["observed_lowest_price_num"] = np.nan
            pred["observed_lowest_price_num"] = pd.to_numeric(pred["observed_lowest_price_num"], errors="coerce")

            if "observed_lowest_dt" not in pred.columns:
                pred["observed_lowest_dt"] = ""

            def _parse_iso(s):
                try:
                    return pd.to_datetime(s, errors="coerce")
                except Exception:
                    return pd.NaT

            pred["observed_lowest_dt_parsed"] = pred["observed_lowest_dt"].apply(_parse_iso)

            if "event_id" not in pred.columns:
                raise KeyError("Predictions must contain 'event_id'.")

            # Merge snapshots (keeps your current behavior)
            snaps = getattr(self, "snapshots", None)
            if snaps is not None and "event_id" in snaps.columns:
                merged = pred.merge(snaps, on="event_id", how="left", suffixes=("", "_snap"))
                if "startDateEastern" not in merged and "startDateEastern_snap" in merged:
                    merged["startDateEastern"] = merged["startDateEastern_snap"]
            else:
                merged = pred

        # Final cleanups
        if "startDateEastern" in merged.columns:
            merged["startDateEastern"] = pd.to_datetime(merged["startDateEastern"], errors="coerce")
        if "week" in merged.columns:
            merged["week"] = pd.to_numeric(merged["week"], errors="coerce").astype("Int64")
        for tcol in ("homeTeam", "awayTeam"):
            if tcol not in merged.columns:
                merged[tcol] = pd.NA
        if "predicted_lowest_price" not in merged.columns and "predicted_lowest_price_num" in merged.columns:
            merged["predicted_lowest_price"] = pd.to_numeric(merged["predicted_lowest_price_num"], errors="coerce")
        merged = merged.dropna(subset=["predicted_lowest_price"])
        return merged


    # ---------------- UI ----------------
    def init_ui(self):
        self.central_widget = QWidget(); self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(); self.central_widget.setLayout(self.layout)

        container = QWidget(); container_layout = QVBoxLayout(); container.setLayout(container_layout)
        container.setStyleSheet("""
            background-color: white; border-radius: 12px; padding: 20px; margin: 16px; border: 1px solid #ccc;
        """)
        container_layout.setSpacing(8); self.layout.addWidget(container)

        title = QLabel("🎟️ Ticket Price Predictor"); title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("QLabel { font-size: 26px; font-weight: bold; margin-bottom: 6px; }")
        container_layout.addWidget(title)

        # Home
        self.home_combo = QComboBox(); self.home_combo.setView(QListView())
        self.home_combo.addItem("-- Select Home Team --")
        homes = sorted(self.df["homeTeam"].dropna().unique()); self.home_combo.addItems(homes)
        self.home_combo.currentIndexChanged.connect(self.update_away_teams)
        self.home_combo.setStyleSheet("QComboBox { font-size: 14px; padding: 4px; border: 1px solid #ccc; border-radius: 4px; background: white; }")
        container_layout.addWidget(self.home_combo)

        # Away
        self.away_combo = QComboBox(); self.away_combo.setView(QListView())
        self.away_combo.addItem("-- Select Away Team --")
        self.away_combo.setStyleSheet("QComboBox { font-size: 14px; padding: 4px; border: 1px solid #ccc; border-radius: 4px; background: white; }")
        container_layout.addWidget(self.away_combo)

        # Predict
        self.predict_button = QPushButton("Get Prediction"); self.predict_button.clicked.connect(self.get_prediction)
        self.predict_button.setStyleSheet("""
            QPushButton { background-color: #1a73e8; color: white; padding: 8px 10px; font-size: 14px; font-weight: bold; border: none; border-radius: 6px; }
            QPushButton:hover { background-color: #1669c1; }
        """); container_layout.addWidget(self.predict_button)

        # Details (scrollable; at least as tall as chart)
        self.details_label = QLabel("")
        self.details_label.setTextFormat(Qt.RichText)
        self.details_label.setWordWrap(True)
        self.details_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.details_label.setStyleSheet(
            "QLabel { background-color: #fafafa; border: 1px solid #ddd; padding: 12px; font-size: 15px; border-radius: 8px; }"
        )
        self.details_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.details_scroll = QScrollArea()
        self.details_scroll.setWidget(self.details_label)
        self.details_scroll.setWidgetResizable(True)
        self.details_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.details_scroll.setStyleSheet("QScrollArea { border: 0; }")
        container_layout.addWidget(self.details_scroll, stretch=0)

        # Countdown
        self.countdown_label = QLabel(""); self.countdown_label.setTextFormat(Qt.RichText); self.countdown_label.setAlignment(Qt.AlignLeft)
        self.countdown_label.setStyleSheet("QLabel { font-size: 14px; padding: 2px 0 8px 2px; }")
        container_layout.addWidget(self.countdown_label, stretch=0)

        # Chart
        self.chart_canvas = FigureCanvas(Figure(constrained_layout=True))
        self.chart_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.chart_canvas.setMinimumHeight(340)
        self.ax = self.chart_canvas.figure.add_subplot(111)
        container_layout.addWidget(self.chart_canvas, stretch=1)
        self.details_scroll.setMinimumHeight(self.chart_canvas.minimumHeight())

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

    def _format_kickoff(self, row) -> str:
        for cand in ["kickoffTimeStr", "kickoff_time", "time_local"]:
            if cand in row and pd.notna(row[cand]) and str(row[cand]).strip():
                t = self._parse_time_hhmm(str(row[cand]))
                if t and not (t.hour == 3 and t.minute == 0):
                    return datetime.strptime(f"{t.hour:02d}:{t.minute:02d}", "%H:%M").strftime("%I:%M %p").lstrip("0")
                return "TBD"
        if "startDateEastern" in row and pd.notna(row["startDateEastern"]):
            t = pd.to_datetime(row["startDateEastern"]).time()
            if t and not (t.hour == 3 and t.minute == 0):
                return datetime.strptime(f"{t.hour:02d}:{t.minute:02d}", "%H:%M").strftime("%I:%M %p").lstrip("0")
        return "TBD"

    def _fmt_money(self, x) -> str:
        try:
            return f"${float(x):,.2f}"
        except Exception:
            return "—"

    def _fmt_dt(self, ts_like) -> str:
        try:
            ts = pd.to_datetime(ts_like)
            if pd.isna(ts):
                return "—"
            return ts.strftime("%A, %B %d, %Y %I:%M %p").lstrip("0").replace(" 0", " ")
        except Exception:
            return "—"

    # ---------------- Prediction & rendering ----------------
    def get_prediction(self):
        try:
            home = self.home_combo.currentText(); away = self.away_combo.currentText()
            if home.startswith("--") or away.startswith("--"):
                self.details_label.setText("<b>❌ Please select both a home and away team.</b>"); self.countdown_label.setText(""); return

            match = self.df[(self.df["homeTeam"] == home) & (self.df["awayTeam"] == away)].sort_values("startDateEastern")
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
        if self.traj_times and self.traj_prices:
            idx = int(np.nanargmin(pd.to_numeric(pd.Series(self.traj_prices), errors="coerce")))
            ts_min = pd.to_datetime(self.traj_times[idx]); p_min = float(self.traj_prices[idx])
            traj_min_txt = f"${p_min:,.2f} on {ts_min.strftime('%a, %b %d, %Y %I:%M %p')}"
        self.details_label.setText(self._details_html(row, traj_min_txt))

    def _details_html(self, row, forecast_min_text: str) -> str:
        game_dt = pd.to_datetime(row["startDateEastern"])
        stadium = row.get("stadium", "Unknown Venue")
        kickoff = self._format_kickoff(row)
        opt_time_str = self._fmt_ampm_no_sec(row.get("optimal_purchase_time"))
        opt_date_iso = row.get("optimal_purchase_date", "")
        opt_date_fmt = self._fmt_full_date(opt_date_iso)
        week_str = int(row["week"]) if pd.notna(row.get("week")) else "—"

        # --- Predicted from model (CSV fields) ---
        predicted_price = row.get("predicted_lowest_price", np.nan)

        # --- Observed (ever) from CSV ---
        observed_price = row.get("observed_lowest_price_num", np.nan)
        # try parsed; fallback to raw string
        obs_ts = row.get("observed_lowest_dt_parsed", pd.NaT)
        if pd.isna(obs_ts) and "observed_lowest_dt" in row:
            try:
                obs_ts = pd.to_datetime(row.get("observed_lowest_dt"), errors="coerce")
            except Exception:
                obs_ts = pd.NaT

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

        # Compute observed lowest price up to kickoff from snapshots (fallbacks included)
        min_price_txt = "—"
        try:
            df = self.snapshots.copy()
            if "event_id" in df.columns:
                df = df[df["event_id"] == event_id]
            if not df.empty:
                df["collected_dt"]  = pd.to_datetime(df.get("collected_dt"), errors="coerce")
                df["lowest_price"]  = pd.to_numeric(df.get("lowest_price"), errors="coerce")
                if df["collected_dt"].notna().any():
                    df = df[df["collected_dt"] <= kickoff] if pd.notna(kickoff) else df
                mp = df["lowest_price"].min()
                if pd.notna(mp):
                    min_price_txt = f"${float(mp):,.2f}"
        except Exception:
            pass

        # Fallbacks from merged row if snapshots were missing
        if min_price_txt == "—":
            for k in ("actual_lowest_price", "observed_lowest_price", "final_lowest_price"):
                if k in row and pd.notna(row[k]):
                    try:
                        min_price_txt = f"${float(row[k]):,.2f}"; break
                    except Exception:
                        continue

        matchup = f"{row.get('homeTeam','?')} vs {row.get('awayTeam','?')}"
        date_str = pd.to_datetime(row.get("startDateEastern")).strftime("%A, %B %d, %Y") if pd.notna(row.get("startDateEastern")) else "Game date unknown"

        ax.text(0.5, 0.68, f"{matchup}", ha="center", va="center", fontsize=16, fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.54, f"Game Date: {date_str}", ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.text(0.5, 0.40, "Observed Lowest Price", ha="center", va="center", fontsize=13, color="#555", transform=ax.transAxes)
        ax.text(0.5, 0.30, min_price_txt, ha="center", va="center", fontsize=20, fontweight="bold", transform=ax.transAxes, color="#2e7d32")

        self.details_label.setText(self._details_html(row, forecast_min_text="—") + "<div style='margin-top:8px;color:#444;'>Game has concluded.</div>")
        self.chart_canvas.draw_idle()

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
