from __future__ import annotations
from typing import Dict, Any
import json

def _minify(s: str) -> str:
    try:
        obj = json.loads(s)
        return json.dumps(obj, separators=(",",":"), sort_keys=True)
    except Exception:
        return ""

def judge_json(task_prompt: str, env: Dict[str, Any]) -> Dict[str, Any]:
    ct = (env.get("final_solution") or {}).get("canonical_text","")
    minified = _minify(ct)
    ok = bool(minified)
    return {"passes_judge": ok, "kind":"json", "minified": minified[:120]}
