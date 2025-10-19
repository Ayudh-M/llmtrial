
from __future__ import annotations
import json, pathlib
from typing import Dict, Any

def load_roleset(path: str) -> Dict[str, Any]:
    p = pathlib.Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
