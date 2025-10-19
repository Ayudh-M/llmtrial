from __future__ import annotations
from typing import Dict, Any, List
import re, json

def _extract_examples(task_prompt: str) -> (List[str], List[str]):
    # very loose: look for JSON-like positives/negatives arrays
    try:
        obj = json.loads(task_prompt)
        return obj.get("positives",[]), obj.get("negatives",[])
    except Exception:
        # naive line-based parsing
        pos, neg = [], []
        lines = task_prompt.splitlines()
        mode = None
        for ln in lines:
            l=ln.strip()
            if l.lower().startswith("positives"):
                mode="pos"; continue
            if l.lower().startswith("negatives"):
                mode="neg"; continue
            if l and mode=="pos":
                pos.append(l.strip(" ,"))
            if l and mode=="neg":
                neg.append(l.strip(" ,"))
        return pos, neg

def judge_regex(task_prompt: str, env: Dict[str, Any]) -> Dict[str, Any]:
    pattern = (env.get("final_solution") or {}).get("canonical_text","")
    try:
        rx = re.compile(pattern)
    except Exception as e:
        return {"passes_judge": False, "kind":"regex", "error": str(e)}
    pos, neg = _extract_examples(task_prompt)
    ok = True
    fails = {"pos_fail":[], "neg_fail":[]}
    for s in pos:
        if rx.search(s) is None:
            ok=False; fails["pos_fail"].append(s)
    for s in neg:
        if rx.search(s) is not None:
            ok=False; fails["neg_fail"].append(s)
    return {"passes_judge": ok, "kind":"regex", "details": fails}
