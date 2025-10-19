from __future__ import annotations
import json, re
from decimal import Decimal
import sqlparse

def canonicalize_for_hash(text: str, kind: str | None = None) -> str:
    s = text.strip()
    if kind == "json":
        try:
            return json.dumps(json.loads(s), separators=(",", ":"), ensure_ascii=False)
        except Exception:
            pass
    if kind == "regex":
        return re.sub(r"\s+", "", s)
    if kind == "sql":
        try:
            return " ".join(sqlparse.format(s, keyword_case="upper", strip_comments=True).split())
        except Exception:
            return " ".join(s.split())
    if kind == "number":
        try:
            return str(Decimal(s).normalize())
        except Exception:
            return s
    # default: collapse whitespace
    return " ".join(s.split())
