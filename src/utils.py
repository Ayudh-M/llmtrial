from __future__ import annotations
import json, hashlib

def to_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
