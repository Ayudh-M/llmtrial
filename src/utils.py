from __future__ import annotations
import json, hashlib, unicodedata

def to_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Cf")
