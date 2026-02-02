from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

_STATUS_PATH = Path("data") / "permanent" / "pipeline_status.json"


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_status(
    name: str,
    status: str,
    detail: str = "",
    extra: dict[str, Any] | None = None,
    path: Path | None = None,
) -> None:
    target = path or _STATUS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    if target.exists():
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    payload[name] = {
        "status": status,
        "detail": detail,
        "extra": extra or {},
        "updated_at": _now_iso(),
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_status(path: Path | None = None) -> dict[str, Any]:
    target = path or _STATUS_PATH
    if not target.exists():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
