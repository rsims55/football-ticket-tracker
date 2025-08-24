# src/gui/ticket_predictor_gui.py
import sys
import os
import pandas as pd
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QComboBox, QPushButton, QMessageBox, QListView, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator

PRED_PATH = "data/predicted/predicted_prices_optimal.csv"
SNAPSHOT_PATH = "data/daily/price_snapshots.csv"

class TicketApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎟️ Ticket Price Predictor")
        self.setGeometry(200, 200, 1000, 720)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f2f5;
                font-family: Arial;
            }
        """)

        self.snapshots = self.load_snapshot_data()
        self.df = self.load_and_merge_data()   # predictions + snapshots (by event_id)

        # state for live countdown / chart re-render
        self.current_row = None
        self.current_event_id = None
        self.ax = None  # cached axes for smoother resize redraws

        self.init_ui()

        # Live countdown timer (1s) — only updates countdown label, not whole details
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_countdown_live)
        self.timer.start(1000)

    # ---------------- Data loading ----------------
    def load_snapshot_data(self) -> pd.DataFrame:
        if not os.path.exists(SNAPSHOT_PATH):
            raise FileNotFoundError(f"Could not find snapshot CSV at '{SNAPSHOT_PATH}'")
        snaps = pd.read_csv(SNAPSHOT_PATH)

        if {"date_collected", "time_collected"}.issubset(snaps.columns):
            snaps["collected_dt"] = pd.to_datetime(
                snaps["date_collected"].astype(str) + " " + snaps["time_collected"].astype(str),
                errors="coerce"
            )
        else:
            snaps["collected_dt"] = pd.NaT

        if "date_local" in snaps.columns:
            snaps["startDateEastern"] = pd.to_datetime(snaps["date_local"], errors="coerce")
        if "homeTeam" not in snaps.columns and "home_team_guess" in snaps.columns:
            snaps = snaps.rename(columns={"home_team_guess": "homeTeam"})
        if "awayTeam" not in snaps.columns and "away_team_guess" in snaps.columns:
            snaps = snaps.rename(columns={"away_team_guess": "awayTeam"})
        return snaps

    def load_and_merge_data(self) -> pd.DataFrame:
        if not os.path.exists(PRED_PATH):
            raise FileNotFoundError(f"Could not find predictions CSV at '{PRED_PATH}'")
        pred = pd.read_csv(PRED_PATH)

        if "startDateEastern" in pred.columns:
            pred["startDateEastern"] = pd.to_datetime(pred["startDateEastern"], errors="coerce")

        if "event_id" not in pred.columns or "event_id" not in self.snapshots.columns:
            raise KeyError("Both predictions and snapshots must contain 'event_id' to merge.")

        merged = pred.merge(
            self.snapshots,
            on="event_id",
            how="left",
            suffixes=("", "_snap")
        )
        if "startDateEastern" not in merged and "startDateEastern_snap" in merged:
            merged["startDateEastern"] = merged["startDateEastern_snap"]

        if "week" in merged.columns:
            merged["week"] = pd.to_numeric(merged["week"], errors="coerce").astype("Int64")
        merged = merged.dropna(subset=["predicted_lowest_price"])

        for tcol in ("homeTeam", "awayTeam"):
            if tcol not in merged.columns:
                merged[tcol] = pd.NA
        return merged

    # ---------------- UI ----------------
    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        container = QWidget()
        container_layout = QVBoxLayout()
        container.setLayout(container_layout)
        container.setStyleSheet("""
            background-color: white;
            border-radius: 12px;
            padding: 20px;
            margin: 16px;
            border: 1px solid #ccc;
        """)
        container_layout.setSpacing(8)
        self.layout.addWidget(container)

        title = QLabel("🎟️ Ticket Price Predictor")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 26px;
                font-weight: bold;
                margin-bottom: 6px;
            }
        """)
        container_layout.addWidget(title)

        # Home dropdown
        self.home_combo = QComboBox()
        self.home_combo.setView(QListView())
        self.home_combo.addItem("-- Select Home Team --")
        homes = sorted(self.df["homeTeam"].dropna().unique())
        self.home_combo.addItems(homes)
        self.home_combo.currentIndexChanged.connect(self.update_away_teams)
        self.home_combo.setStyleSheet("""
            QComboBox { font-size: 14px; padding: 4px; border: 1px solid #ccc; border-radius: 4px; background: white; }
            QComboBox QAbstractItemView::item { padding: 6px 8px; }
            QListView::item:hover { color: #1a73e8; }
        """)
        container_layout.addWidget(self.home_combo)

        # Away dropdown
        self.away_combo = QComboBox()
        self.away_combo.setView(QListView())
        self.away_combo.addItem("-- Select Away Team --")
        self.away_combo.setStyleSheet("""
            QComboBox { font-size: 14px; padding: 4px; border: 1px solid #ccc; border-radius: 4px; background: white; }
            QComboBox QAbstractItemView::item { padding: 6px 8px; }
            QListView::item:hover { color: #1a73e8; }
        """)
        container_layout.addWidget(self.away_combo)

        # Predict button
        self.predict_button = QPushButton("Get Prediction")
        self.predict_button.clicked.connect(self.get_prediction)
        self.predict_button.setStyleSheet("""
            QPushButton {
                background-color: #1a73e8; color: white; padding: 8px 10px;
                font-size: 14px; font-weight: bold; border: none; border-radius: 6px;
            }
            QPushButton:hover { background-color: #1669c1; }
        """)
        container_layout.addWidget(self.predict_button)

        # --- Details area ---
        self.details_label = QLabel("")
        self.details_label.setTextFormat(Qt.RichText)
        self.details_label.setWordWrap(True)
        self.details_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.details_label.setStyleSheet("""
            QLabel {
                background-color: #fafafa; border: 1px solid #ddd; padding: 12px;
                font-size: 15px; border-radius: 8px;
            }
        """)
        self.details_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        container_layout.addWidget(self.details_label, stretch=0)

        # Countdown
        self.countdown_label = QLabel("")
        self.countdown_label.setTextFormat(Qt.RichText)
        self.countdown_label.setAlignment(Qt.AlignLeft)
        self.countdown_label.setStyleSheet("""
            QLabel { font-size: 14px; padding: 2px 0 8px 2px; }
        """)
        container_layout.addWidget(self.countdown_label, stretch=0)

        # ---- Chart (responsive) ----
        # Use constrained_layout so margins/ticks adapt on resize
        self.chart_canvas = FigureCanvas(Figure(constrained_layout=True))
        self.chart_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.chart_canvas.setMinimumHeight(260)
        self.ax = self.chart_canvas.figure.add_subplot(111)
        container_layout.addWidget(self.chart_canvas, stretch=1)  # canvas takes remaining vertical space

    # ----- Window resize: refresh chart ticks and redraw efficiently -----
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_event_id is not None:
            self.render_chart(self.current_event_id, reuse_axes=True)
        else:
            self.chart_canvas.draw_idle()

    # ---------------- Dropdown logic ----------------
    def update_away_teams(self):
        selected_home = self.home_combo.currentText()
        if selected_home.startswith("--"):
            self.away_combo.clear()
            self.away_combo.addItem("-- Select Away Team --")
            return

        df_home = self.df[self.df["homeTeam"] == selected_home].sort_values("startDateEastern")
        seen = set()
        ordered_away = []
        for away in df_home["awayTeam"]:
            if pd.isna(away):
                continue
            if away not in seen:
                ordered_away.append(away)
                seen.add(away)

        self.away_combo.clear()
        self.away_combo.addItem("-- Select Away Team --")
        self.away_combo.addItems(ordered_away)

    # ---------------- Formatting helpers ----------------
    def _fmt_ampm_no_sec(self, val: str) -> str:
        if val is None or str(val).strip() == "" or str(val).strip().upper() == "TBD":
            return "TBD"
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
            dt = pd.to_datetime(dt_like)
            return dt.strftime("%A, %B %d, %Y")
        except Exception:
            return "TBD"

    def _parse_time_hhmm(self, t_str: str):
        if not t_str:
            return None
        try:
            return pd.to_datetime(str(t_str)).time()
        except Exception:
            for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M%p"):
                try:
                    return datetime.strptime(str(t_str), fmt).time()
                except Exception:
                    continue
            return None

    def _humanize_delta(self, td: pd.Timedelta) -> str:
        total_seconds = int(td.total_seconds())
        sign = "-" if total_seconds < 0 else ""
        total_seconds = abs(total_seconds)
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        parts = []
        if days: parts.append(f"{days}d")
        if hours or days: parts.append(f"{hours}h")
        parts.append(f"{minutes}m")
        parts.append(f"{seconds}s")
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

    # ---------------- Prediction / Rendering ----------------
    def get_prediction(self):
        try:
            home = self.home_combo.currentText()
            away = self.away_combo.currentText()

            if home.startswith("--") or away.startswith("--"):
                self.details_label.setText("<b>❌ Please select both a home and away team.</b>")
                self.countdown_label.setText("")
                return

            match = self.df[
                (self.df["homeTeam"] == home) &
                (self.df["awayTeam"] == away)
            ].sort_values("startDateEastern")

            if match.empty:
                self.details_label.setText("<b>❌ No prediction found for this matchup.</b>")
                self.countdown_label.setText("")
                return

            now = pd.Timestamp.now()
            upcoming = match[match["startDateEastern"] >= now]
            row = (upcoming.iloc[0] if not upcoming.empty else match.iloc[0]).to_dict()

            self.current_row = row
            self.current_event_id = row.get("event_id")
            self.render_details(row)          # static info once
            self.update_countdown_live()      # live piece now
            self.render_chart(self.current_event_id, reuse_axes=False)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not fetch prediction.\n{e}")

    def render_details(self, row):
        game_dt = pd.to_datetime(row["startDateEastern"])
        stadium = row.get("stadium", "Unknown Venue")
        kickoff = self._format_kickoff(row)

        # Optimal purchase strings (static parts)
        opt_time_str = self._fmt_ampm_no_sec(row.get("optimal_purchase_time"))
        opt_date_iso = row.get("optimal_purchase_date", "")
        opt_date_fmt = self._fmt_full_date(opt_date_iso)

        week_str = int(row["week"]) if pd.notna(row.get("week")) else "—"

        html = f"""
            <div style="font-size: 15px; line-height: 1.6;">
                <h2 style="margin-bottom: 2px;">{row['homeTeam']} vs {row['awayTeam']}</h2>
                <div style="font-size: 18px; font-weight: 700; color: #6a1b9a; margin-bottom: 6px;">
                    Predicted Price: ${row['predicted_lowest_price']:.2f}
                </div><br>

                <b>Optimal Purchase Date:</b> {opt_date_fmt}<br>
                <b>Optimal Purchase Time:</b> {opt_time_str}<br>

                <b>Game Week:</b> {week_str}<br>
                <b>Game Date:</b> {game_dt.strftime('%A, %B %d, %Y')}<br>
                <b>Kickoff Time:</b> {kickoff}<br>
                <b>Venue:</b> {stadium}<br>
            </div>
        """
        self.details_label.setText(html)

    def _tick_step_for_height(self) -> int:
        """Use finer price tick steps on larger chart heights."""
        h = self.chart_canvas.height()
        if h >= 700:
            return 20
        if h >= 600:
            return 25
        if h >= 480:
            return 50
        return 100

    def render_chart(self, event_id, reuse_axes: bool = True):
        if reuse_axes and self.ax is not None:
            self.ax.clear()
            ax = self.ax
        else:
            self.chart_canvas.figure.clear()
            ax = self.chart_canvas.figure.add_subplot(111)
            self.ax = ax  # cache for future resizes

        snap_filtered = self.snapshots[self.snapshots["event_id"] == event_id].dropna(
            subset=["collected_dt", "lowest_price"], how="any"
        )

        if snap_filtered.empty:
            ax.set_title("No snapshot data available")
        else:
            snap_filtered = snap_filtered.sort_values("collected_dt")
            ax.plot(snap_filtered["collected_dt"], snap_filtered["lowest_price"], label="Lowest Price", linewidth=2)

            ax.set_title("Price Trend Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price ($)")
            ax.legend()

            # adaptive date axis
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M'))
            ax.tick_params(axis='x', rotation=45)

            # dynamic Y-axis tick step
            try:
                ax.yaxis.set_major_locator(MultipleLocator(self._tick_step_for_height()))
            except Exception:
                pass

        # with constrained_layout=True, margins adapt, but add a touch more bottom padding
        self.chart_canvas.figure.subplots_adjust(bottom=0.18)

        # coalesced redraw (smoother during resizes)
        self.chart_canvas.draw_idle()

    # ---------------- Live countdown tick ----------------
    def update_countdown_live(self):
        """Refresh ONLY the countdown label every 1s; avoids scroll/resize jumps."""
        if not self.current_row:
            return

        opt_date_iso = self.current_row.get("optimal_purchase_date", "")
        parsed_time = self._parse_time_hhmm(self.current_row.get("optimal_purchase_time"))

        if not opt_date_iso or not parsed_time:
            self.countdown_label.setText("")
            return

        opt_dt = pd.to_datetime(f"{opt_date_iso} {parsed_time.strftime('%H:%M:%S')}", errors="coerce")
        if pd.isna(opt_dt):
            self.countdown_label.setText("")
            return

        now_dt = pd.Timestamp.now()
        delta = opt_dt - now_dt
        human = self._humanize_delta(delta)
        if delta.total_seconds() < 0:
            html = '<div style="font-weight:700; color:#c62828;">PAST OPTIMAL DATE</div>'
        else:
            html = f'<div style="color:#2e7d32;">Countdown to Optimal Ticket Price: {human}</div>'
        self.countdown_label.setText(html)

def main():
    """Entry point for cfb-tix-gui."""
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    win = TicketApp()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TicketApp()
    window.show()
    sys.exit(app.exec_())
