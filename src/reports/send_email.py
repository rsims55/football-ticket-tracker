#!/usr/bin/env python3
import os
import re
import glob
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Try to import markdown -> HTML converter
try:
    import markdown as md
    def md_to_html(text: str) -> str:
        return md.markdown(
            text,
            extensions=["extra", "sane_lists", "toc", "nl2br", "smarty"],
            output_format="html5",
        )
except Exception:
    def md_to_html(text: str) -> str:
        # Minimal fallback
        lines = text.splitlines()
        html_lines = []
        for ln in lines:
            if ln.startswith("### "):
                html_lines.append(f"<h3>{ln[4:]}</h3>")
            elif ln.startswith("## "):
                html_lines.append(f"<h2>{ln[3:]}</h2>")
            elif ln.startswith("# "):
                html_lines.append(f"<h1>{ln[2:]}</h1>")
            elif ln.strip() == "":
                html_lines.append("<br>")
            else:
                html_lines.append(f"<p>{ln}</p>")
        return "\n".join(html_lines)

# Required env vars
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
TO_EMAIL = os.getenv("TO_EMAIL")
APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

def _fail(msg: str) -> None:
    print(f"❌ {msg}")
    raise SystemExit(1)

def _validate_env():
    missing = [name for name, val in {
        "GMAIL_ADDRESS": GMAIL_ADDRESS,
        "TO_EMAIL": TO_EMAIL,
        "GMAIL_APP_PASSWORD": APP_PASSWORD,
    }.items() if not val]
    if missing:
        hint = (
            "Set these env vars (PowerShell examples):\n"
            "  $env:GMAIL_ADDRESS = 'you@gmail.com'\n"
            "  $env:TO_EMAIL = 'you@gmail.com'\n"
            "  $env:GMAIL_APP_PASSWORD = '<your 16-char app password>'"
        )
        _fail(f"Missing env vars: {', '.join(missing)}\n\n{hint}")

def _find_report_path() -> str | None:
    today = datetime.now().strftime("%Y-%m-%d")
    preferred = os.path.join("reports", "weekly", today, f"weekly_report_{today}.md")
    if os.path.exists(preferred):
        return preferred
    pattern = os.path.join("reports", "weekly", "*", "weekly_report_*.md")
    matches = glob.glob(pattern)
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]

def _wrap_html(body_html: str, title: str) -> str:
    # Inline CSS + legacy attrs help Outlook/Gmail render borders
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; line-height: 1.4; }}
  h1,h2,h3 {{ margin: 0.6em 0 0.3em; }}
  code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  pre {{ background: #f8f8f8; padding: 10px; overflow-x: auto; }}
  .small {{ color: #666; font-size: 12px; margin-top: 16px; }}
  /* Fallback if attributes are stripped */
  table, th, td {{ border: 1px solid #cccccc; border-collapse: collapse; }}
  th, td {{ padding: 6px 8px; }}
  th {{ background: #f5f5f5; }}
  img {{ max-width: 100%; height: auto; }}
</style>
</head>
<body>
{body_html}
<p class="small">Sent automatically by the Weekly Ticket Model pipeline.</p>
</body>
</html>"""

def _force_table_borders(html: str) -> str:
    # Add legacy attributes (Outlook loves these) if missing
    html = re.sub(
        r"<table(?![^>]*\bborder=)",
        r'<table border="1" cellpadding="6" cellspacing="0"',
        html,
        flags=re.IGNORECASE,
    )
    # Ensure th/td have padding via inline style (in case CSS is stripped)
    def _pad_cell(m):
        tag = m.group(1)
        attrs = m.group(2) or ""
        if "style=" in attrs.lower():
            return f"<{tag}{attrs}>"
        return f'<{tag}{attrs} style="padding:6px 8px; border:1px solid #cccccc;">'
    html = re.sub(r"<(th)([^>]*)>", _pad_cell, html, flags=re.IGNORECASE)
    html = re.sub(r"<(td)([^>]*)>", _pad_cell, html, flags=re.IGNORECASE)
    return html

def _embed_images(html: str, report_dir: str):
    """
    Replace <img src="..."> with cid: links and prepare MIMEImage parts.
    Looks for images relative to the report dir (e.g., images/plot.png).
    """
    cid_parts = []
    used = {}

    def repl(match):
        src = match.group(1).strip()
        # Ignore absolute http(s) images; handle only local relative paths
        if re.match(r"(?i)^https?://", src):
            return match.group(0)

        # Resolve relative path against the report directory
        img_path = os.path.normpath(os.path.join(report_dir, src))
        if not os.path.exists(img_path):
            # Try common report subfolder 'images/'
            fallback = os.path.join(report_dir, "images", os.path.basename(src))
            if os.path.exists(fallback):
                img_path = fallback
            else:
                # leave as-is if not found
                return match.group(0)

        # Reuse CID if the same file appears multiple times
        if img_path in used:
            cid = used[img_path]
        else:
            cid = f"img{len(used)+1}@cfb-report"
            used[img_path] = cid
            with open(img_path, "rb") as f:
                data = f.read()
            mime = MIMEImage(data)
            mime.add_header("Content-ID", f"<{cid}>")
            mime.add_header("Content-Disposition", "inline", filename=os.path.basename(img_path))
            cid_parts.append(mime)

        return f'<img src="cid:{cid}" alt="{os.path.basename(img_path)}">'

    # Replace src in <img ...>
    new_html = re.sub(r'<img[^>]*\bsrc=["\']([^"\']+)["\'][^>]*>', repl, html, flags=re.IGNORECASE)
    return new_html, cid_parts

def send_report(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        md_text = f.read()

    subject_date = datetime.now().strftime("%Y-%m-%d")
    subject = f"📬 Weekly Ticket Model Report – {subject_date}"

    # Convert Markdown to HTML
    html_core = md_to_html(md_text)

    # Force robust table rendering
    html_core = _force_table_borders(html_core)

    # Wrap with head/body and CSS
    html_body = _wrap_html(html_core, subject)

    # Embed images (cid) relative to report dir
    report_dir = os.path.dirname(filepath)
    html_body_with_images, image_parts = _embed_images(html_body, report_dir)

    # Build MIME structure:
    # multipart/related
    #   └── multipart/alternative
    #         ├── text/plain  (markdown as fallback)
    #         └── text/html    (our HTML)
    #   └── [inline images...]
    outer = MIMEMultipart("related")
    outer["Subject"] = subject
    outer["From"] = GMAIL_ADDRESS
    outer["To"] = TO_EMAIL

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(md_text, "plain", "utf-8"))
    alt.attach(MIMEText(html_body_with_images, "html", "utf-8"))
    outer.attach(alt)

    # Attach inline images
    for img_part in image_parts:
        outer.attach(img_part)

    # Attach the original markdown
    attachment = MIMEBase("text", "markdown")
    attachment.set_payload(md_text.encode("utf-8"))
    encoders.encode_base64(attachment)
    attachment.add_header("Content-Disposition", "attachment", filename=os.path.basename(filepath))
    outer.attach(attachment)

    # Send
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_ADDRESS, APP_PASSWORD)
            server.send_message(outer)
        print(f"📧 Email sent to {TO_EMAIL}")
        print(f"🗂️  Report file: {filepath}")
    except smtplib.SMTPAuthenticationError:
        _fail("Gmail authentication failed. Check GMAIL_ADDRESS and GMAIL_APP_PASSWORD "
              "(must be a Gmail **App Password** with 2-Step Verification).")
    except smtplib.SMTPResponseException as e:
        _fail(f"SMTP error {e.smtp_code}: {e.smtp_error!r}")
    except Exception as e:
        _fail(f"Unexpected error sending email: {e}")

if __name__ == "__main__":
    _validate_env()
    report_path = _find_report_path()
    if not report_path:
        _fail("Report file not found under reports/weekly/*/weekly_report_*.md. "
              "Run generate_weekly_report.py first.")
    send_report(report_path)
