from __future__ import annotations
from typing import Dict, Any

def judge_boolean(task_prompt: str, env: Dict[str, Any]) -> Dict[str, Any]:
    ct = (env.get("final_solution") or {}).get("canonical_text","").strip().upper()
    ok = ct in {"TRUE","FALSE"}
    return {"passes_judge": ok, "kind":"boolean"}
