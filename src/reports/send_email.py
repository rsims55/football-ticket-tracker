import smtplib
from email.mime.text import MIMEText
import os
from datetime import datetime

GMAIL_ADDRESS = os.getenv("TO_EMAIL")
TO_EMAIL = os.getenv("TO_EMAIL")
APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

def send_report(filepath):
    with open(filepath, "r") as f:
        body = f.read()

    msg = MIMEText(body, "plain")
    msg["Subject"] = f"📬 Weekly Ticket Model Report – {datetime.now().strftime('%Y-%m-%d')}"
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = TO_EMAIL

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, APP_PASSWORD)
        server.send_message(msg)

    print(f"📧 Email sent to {TO_EMAIL}")

if __name__ == "__main__":
    report_path = f"reports/weekly_report_{datetime.now().strftime('%Y-%m-%d')}.md"
    if os.path.exists(report_path):
        send_report(report_path)
    else:
        print("❌ Report file not found. Run generate_weekly_report.py first.")
