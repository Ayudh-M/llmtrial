from __future__ import annotations
from typing import Dict, Any
import re, math

NUM_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?$")

def judge_math(task_prompt: str, env: Dict[str, Any]) -> Dict[str, Any]:
    ct = (env.get("final_solution") or {}).get("canonical_text","").strip()
    # format check only (no ground truth): numeric string
    ok = bool(NUM_RE.match(ct))
    return {"passes_judge": ok, "kind":"format:number"}
