from __future__ import annotations
import json, hashlib, unicodedata

def to_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    """Normalize text with NFKC and strip invisible format characters."""
    normed = unicodedata.normalize("NFKC", text or "")
    return "".join(ch for ch in normed if unicodedata.category(ch) != "Cf")
