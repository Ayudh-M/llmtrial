from __future__ import annotations

"""Helpers for canonicalizing strings prior to hashing."""

import json
import re
from decimal import Decimal, InvalidOperation

import sqlparse


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    return (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    )


def _looks_like_sql(text: str) -> bool:
    lowered = text.strip().lower()
    return "select" in lowered and "from" in lowered


def canonicalize_for_hash(text: str, kind: str | None = None) -> str:
    """Return a canonical textual form suitable for hashing comparisons."""

    s = (text or "").strip()

    detected = kind
    if detected is None:
        if _looks_like_json(s):
            detected = "json"
        elif re.match(r"^(SELECT|INSERT|UPDATE|DELETE)\b", s, flags=re.IGNORECASE):
            detected = "sql"

    kind = detected or kind

    if kind == "json" or (kind is None and _looks_like_json(s)):
        try:
            payload = json.loads(s)
        except Exception:
            pass
        else:
            return json.dumps(payload, separators=(",", ":"), ensure_ascii=False, sort_keys=True)

    if kind == "regex":
        return re.sub(r"\s+", "", s)

    if kind == "sql" or (kind is None and _looks_like_sql(s)):
        try:
            formatted = sqlparse.format(s, keyword_case="upper", strip_comments=True)
        except Exception:
            formatted = s
        cleaned = re.sub(r"/\*.*?\*/", " ", formatted, flags=re.DOTALL)
        cleaned = re.sub(r"--.*?(?=\n|$)", " ", cleaned)
        return " ".join(cleaned.split())

    if kind == "number":
        try:
            value = Decimal(s)
        except (InvalidOperation, ValueError):
            return s.strip()
        normalized = value.normalize()
        return format(normalized, "f").rstrip("0").rstrip(".") if normalized == normalized.to_integral() else format(normalized, "f")

    return " ".join(s.split())
