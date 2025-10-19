
from __future__ import annotations
import os, csv, json, uuid, platform, sys
from datetime import datetime, timezone
from typing import Iterable, Dict, Any

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_jsonl(path: str, lines: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

def append_csv_row(csv_path: str, fieldnames: list[str], row: Dict[str, Any]) -> None:
    # Create if missing with header
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})

def basic_env_info() -> dict:
    try:
        import torch, transformers
    except Exception:
        torch = None
        transformers = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", "n/a"),
        "transformers": getattr(transformers, "__version__", "n/a"),
    }
