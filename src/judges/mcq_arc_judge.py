from __future__ import annotations
from typing import Dict, Any
import re
def judge_mcq_arc(task_prompt: str, env: Dict[str, Any]) -> Dict[str, Any]:
    ct = (env.get("final_solution") or {}).get("canonical_text","").strip().upper()
    ok = ct in {"A","B","C","D"}
    return {"passes_judge": ok, "kind":"mcq_arc"}
