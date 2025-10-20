from __future__ import annotations
import json, re
from decimal import Decimal
import sqlparse

def _looks_like_json(text: str) -> bool:
    t = text.strip()
    return (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]"))


def _looks_like_sql(text: str) -> bool:
    t = text.strip().lower()
    return bool(re.search(r"\bselect\b", t)) and bool(re.search(r"\bfrom\b", t))


def canonicalize_for_hash(text: str, kind: str | None = None) -> str:
    s = text.strip()

    autodetect = kind
    if autodetect is None:
        if s[:1] in "[{":
            autodetect = "json"
        elif re.match(r"^(SELECT|INSERT|UPDATE|DELETE)\b", s, flags=re.IGNORECASE):
            autodetect = "sql"
    kind = autodetect

    if kind == "json" or (kind is None and _looks_like_json(s)):
        try:
            parsed = json.loads(s)
        except Exception:
            pass
        else:
            return json.dumps(parsed, separators=(",", ":"), ensure_ascii=False, sort_keys=True)
    if kind == "regex":
        return re.sub(r"\s+", "", s)
    if kind == "sql" or (kind is None and _looks_like_sql(s)):
        stripped = re.sub(r"/\*.*?\*/", " ", s, flags=re.DOTALL)
        stripped = re.sub(r"--.*?(?=\n|$)", " ", stripped)
        try:
            formatted = sqlparse.format(stripped, keyword_case="upper", strip_comments=True)
        except Exception:
            return " ".join(stripped.split())
        return " ".join(formatted.split())
    if kind == "number":
        try:
            return str(Decimal(s).normalize())
        except Exception:
            return s
    # default: collapse whitespace
    return " ".join(s.split())
