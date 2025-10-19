from __future__ import annotations
from typing import Dict, Any

def judge_headline(task_prompt: str, env: Dict[str, Any]) -> Dict[str, Any]:
    txt = (env.get("final_solution") or {}).get("canonical_text","").strip()
    words = [w for w in txt.split() if w.strip()]
    ok = 1 <= len(words) <= 12
    return {"passes_judge": ok, "kind":"headline", "length": len(words)}
