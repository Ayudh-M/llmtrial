from __future__ import annotations
from typing import Dict, Any

def judge_paraphrase(task_prompt: str, env: Dict[str, Any]) -> Dict[str, Any]:
    # Accept only SAME/DIFFERENT labels
    ct = (env.get("final_solution") or {}).get("canonical_text","").strip().upper()
    ok = ct in {"SAME","DIFFERENT"}
    return {"passes_judge": ok, "kind":"paraphrase_label"}
