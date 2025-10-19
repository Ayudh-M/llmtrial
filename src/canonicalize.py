from __future__ import annotations
import json, re
from decimal import Decimal
import sqlparse

def canonicalize_for_hash(text: str, kind: str | None = None) -> str:
    s = text.strip()

    autodetect = kind
    if autodetect is None:
        if s[:1] in "[{":
            autodetect = "json"
        elif re.match(r"^(SELECT|INSERT|UPDATE|DELETE)\b", s, flags=re.IGNORECASE):
            autodetect = "sql"
    kind = autodetect

    if kind == "json":
        try:
            return json.dumps(
                json.loads(s), separators=(",", ":"), ensure_ascii=False, sort_keys=True
            )
        except Exception:
            pass
    if kind == "regex":
        return re.sub(r"\s+", "", s)
    if kind == "sql":
        try:
            formatted = sqlparse.format(s, keyword_case="upper", strip_comments=True)
            return " ".join(formatted.split())
        except Exception:
            return " ".join(s.split())
    if kind == "number":
        try:
            return str(Decimal(s).normalize())
        except Exception:
            return s
    # default: collapse whitespace
    return " ".join(s.split())
