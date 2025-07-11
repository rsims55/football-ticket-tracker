import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QComboBox, QPushButton, QMessageBox, QTextEdit
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates

PRED_PATH = "data/predicted_prices_optimal.csv"
SCHED_PATH = "data/enriched_schedule_2025.csv"
SNAPSHOT_PATH = "data/price_snapshots.csv"

class TicketApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎟️ Ticket Price Predictor")
        self.setGeometry(200, 200, 800, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f2f5;
                font-family: Arial;
            }
        """)

        self.df = self.load_and_merge_data()
        self.snapshots = self.load_snapshot_data()
        self.init_ui()

    def load_and_merge_data(self):
        pred = pd.read_csv(PRED_PATH)
        sched = pd.read_csv(SCHED_PATH)

        pred["startDateEastern"] = pd.to_datetime(pred["startDateEastern"], errors="coerce", utc=True).dt.tz_convert(None)
        sched["startDateEastern"] = pd.to_datetime(sched["startDateEastern"], errors="coerce", utc=True).dt.tz_convert(None)

        merged = pd.merge(
            pred,
            sched,
            on=["homeTeam", "awayTeam", "startDateEastern"],
            how="left",
            suffixes=("", "_sched")
        )

        merged = merged.dropna(subset=["predicted_lowest_price"])
        merged["week"] = pd.to_numeric(merged["week"], errors="coerce").astype("Int64")
        return merged

    def load_snapshot_data(self):
        snapshots = pd.read_csv(SNAPSHOT_PATH)
        snapshots["collected_dt"] = pd.to_datetime(
            snapshots["date_collected"] + " " + snapshots["time_collected"],
            errors="coerce"
        )
        return snapshots

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Container "card"
        container = QWidget()
        container_layout = QVBoxLayout()
        container.setLayout(container_layout)
        container.setStyleSheet("""
            background-color: white;
            border-radius: 12px;
            padding: 24px;
            margin: 20px;
            border: 1px solid #ccc;
        """)
        container_layout.setSpacing(0)  # Tighter vertical spacing
        self.layout.addWidget(container)

        # Title
        title = QLabel("🎟️ Ticket Price Predictor")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 4px;
            }
        """)
        container_layout.addWidget(title)

        # Home Team Dropdown
        self.home_combo = QComboBox()
        self.home_combo.addItem("-- Select Home Team --")
        home_teams = sorted(self.df["homeTeam"].unique())
        self.home_combo.addItems(home_teams)
        self.home_combo.currentIndexChanged.connect(self.update_away_teams)
        self.home_combo.setStyleSheet("""
            QComboBox {
                font-size: 14px;
                padding: 4px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
            }
        """)
        container_layout.addWidget(self.home_combo)

        # Away Team Dropdown
        self.away_combo = QComboBox()
        self.away_combo.addItem("-- Select Away Team --")
        self.away_combo.setStyleSheet("""
            QComboBox {
                font-size: 14px;
                padding: 4px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
            }
        """)
        container_layout.addWidget(self.away_combo)

        # Button
        self.predict_button = QPushButton("Get Prediction")
        self.predict_button.clicked.connect(self.get_prediction)
        self.predict_button.setStyleSheet("""
            QPushButton {
                background-color: #1a73e8;
                color: white;
                padding: 8px 10px;
                font-size: 14px;
                font-weight: bold;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #1669c1;
            }
        """)
        container_layout.addWidget(self.predict_button)

        # Output Display
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setStyleSheet("""
            background-color: #fafafa;
            border: 1px solid #ddd;
            padding: 14px;
            font-size: 14px;
            border-radius: 8px;
        """)
        container_layout.addWidget(self.output)

        # Chart Canvas
        self.chart_canvas = FigureCanvas(Figure(figsize=(6, 3)))
        container_layout.addWidget(self.chart_canvas)

    def update_away_teams(self):
        selected_home = self.home_combo.currentText()

        matchups = (
            self.df[self.df["homeTeam"] == selected_home]
            .sort_values("startDateEastern")
        )

        seen = set()
        ordered_away_teams = []
        for away in matchups["awayTeam"]:
            if away not in seen:
                ordered_away_teams.append(away)
                seen.add(away)

        self.away_combo.clear()
        self.away_combo.addItem("-- Select Away Team --")
        self.away_combo.addItems(ordered_away_teams)

    def get_prediction(self):
        try:
            home = self.home_combo.currentText()
            away = self.away_combo.currentText()

            if home == "-- Select Home Team --" or away == "-- Select Away Team --":
                self.output.setText("❌ Please select both a home and away team.")
                return

            match = self.df[
                (self.df["homeTeam"] == home) &
                (self.df["awayTeam"] == away)
            ].sort_values("startDateEastern")

            if match.empty:
                self.output.setText("❌ No prediction found for this matchup.")
                return

            row = match.iloc[0]
            game_date = pd.to_datetime(row["startDateEastern"])
            kickoff = row["kickoffTimeStr"] if pd.notna(row["kickoffTimeStr"]) else "TBD"
            stadium = row.get("stadium", "Unknown Venue")
            location = f"{row.get('city', '')}, {row.get('state', '')}".strip(", ")
            if not location or location == ",":
                location = "Unknown Location"

            days_until_game = (game_date.date() - pd.Timestamp.now().date()).days
            countdown_str = f"{days_until_game} day(s) away" if days_until_game >= 0 else "Game has passed"

            # Get most recent snapshot for this matchup
            snap_match = self.snapshots[
                (self.snapshots["homeTeam"] == home) &
                (self.snapshots["awayTeam"] == away)
            ].sort_values("collected_dt", ascending=False)

            if not snap_match.empty:
                latest_snap = snap_match.iloc[0]
                current_low_str = f"${latest_snap['lowest_price']:.2f}" if pd.notna(latest_snap["lowest_price"]) else "N/A"
                current_avg_str = f"${latest_snap['average_price']:.2f}" if pd.notna(latest_snap["average_price"]) else "N/A"
            else:
                current_low_str = "N/A"
                current_avg_str = "N/A"

            # Format HTML display
            text = f"""
                <div style="font-size: 14px; line-height: 1.6;">
                    <h2 style="margin-bottom: 4px;">{home} vs {away}</h2><
                    <b style="font-size: 15px;">Predicted Lowest Price:</b>
                    <span style="font-size: 18px; font-weight: bold; color: #6a1b9a;">
                        ${row['predicted_lowest_price']:.2f}
                    </span><br>
                    <b>Optimal Purchase Date:</b> {row['optimal_purchase_date']}<br>
                    <b>Optimal Purchase Time:</b> {row['optimal_purchase_time']}<br><br>

                    <b>Week:</b> {int(row['week'])}<br>
                    <b>Date:</b> {game_date.strftime('%A, %B %d, %Y')}<br>
                    <b>Kickoff:</b> {kickoff}<br>
                    <b>Venue:</b> {stadium}<br>
                    <b>Location:</b> {location}<br>
                    <b>Countdown:</b> {countdown_str}<br>

                    <hr style="margin-top: 12px; margin-bottom: 12px;">

                    <b style="color: #555;">Current Lowest Price:</b> <span style="font-weight: bold;">{current_low_str}</span><br>
                    <b style="color: #555;">Current Average Price:</b> <span style="font-weight: bold;">{current_avg_str}</span>
                </div>
            """
            self.output.setHtml(text)

            # === Plot chart ===
            self.chart_canvas.figure.clear()
            ax = self.chart_canvas.figure.add_subplot(111)

            snap_filtered = self.snapshots[
                (self.snapshots["homeTeam"] == home) &
                (self.snapshots["awayTeam"] == away)
            ].dropna(subset=["collected_dt", "lowest_price", "average_price"])

            if snap_filtered.empty:
                ax.set_title("No snapshot data available")
            else:
                snap_filtered = snap_filtered.sort_values("collected_dt")

                try:
                    ax.plot(
                        snap_filtered["collected_dt"],
                        snap_filtered["lowest_price"],
                        label="Lowest Price",
                        color="#1a73e8",  # blue
                        linewidth=2
                    )

                    ax.plot(
                        snap_filtered["collected_dt"],
                        snap_filtered["average_price"],
                        label="Average Price",
                        color="#888888",  # gray
                        linewidth=2
                    )

                    ax.set_title("Price Trend Over Time")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Price ($)")
                    ax.legend()

                    # Format x-axis
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                    ax.tick_params(axis='x', rotation=45)

                    # Adjust layout to prevent label cutoff
                    self.chart_canvas.figure.subplots_adjust(bottom=0.4)

                    self.chart_canvas.draw()

                except Exception as plot_err:
                    print("Chart error:", plot_err)
                    ax.set_title("⚠️ Error displaying chart")

            self.chart_canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not fetch prediction.\n{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TicketApp()
    window.show()
    sys.exit(app.exec_())
