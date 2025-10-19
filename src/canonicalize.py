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

    if kind == "json":
        try:
            return json.dumps(
                json.loads(s), separators=(",", ":"), ensure_ascii=False, sort_keys=True
            )
    if kind == "json" or (kind is None and _looks_like_json(s)):
        try:
            return json.dumps(json.loads(s), separators=(",", ":"), ensure_ascii=False, sort_keys=True)
        except Exception:
            pass
    if kind == "regex":
        return re.sub(r"\s+", "", s)
    if kind == "sql" or (kind is None and _looks_like_sql(s)):
        try:
            formatted = sqlparse.format(s, keyword_case="upper", strip_comments=True)
            stripped = re.sub(r"/\*.*?\*/", " ", s, flags=re.DOTALL)
            stripped = re.sub(r"--.*?(?=\n|$)", " ", stripped)
            formatted = sqlparse.format(stripped, keyword_case="upper", strip_comments=True)
            return " ".join(formatted.split())
        except Exception:
            cleaned = re.sub(r"/\*.*?\*/", " ", s)
            cleaned = re.sub(r"--.*?(?=\n|$)", " ", cleaned)
            return " ".join(cleaned.split())
    if kind == "number":
        try:
            return str(Decimal(s).normalize())
        except Exception:
            return s
    # default: collapse whitespace
    return " ".join(s.split())
