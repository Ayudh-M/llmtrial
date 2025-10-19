from __future__ import annotations
from typing import Dict, Any
from .math_judge import judge_math
from .regex_judge import judge_regex
from .sql_judge import judge_sql
from .json_judge import judge_json
from .boolean_judge import judge_boolean
from .headline_xsum_judge import judge_headline
from .paraphrase_judge import judge_paraphrase
from .mcq_arc_judge import judge_mcq_arc
from .winogrande_judge import judge_winogrande

REGISTRY = {
    "math": judge_math,
    "regex": judge_regex,
    "sql": judge_sql,
    "json": judge_json,
    "boolean": judge_boolean,
    "headline_xsum": judge_headline,
    "paraphrase": judge_paraphrase,
    "mcq_arc": judge_mcq_arc,
    "winogrande": judge_winogrande,
}

def judge_auto(task_prompt: str, envelope: Dict[str, Any], roleset_id: str | None = None) -> Dict[str, Any]:
    # crude heuristics based on roleset_id or prompt patterns
    ct = (envelope.get("final_solution") or {}).get("canonical_text","") or ""
    if not ct:
        return {"passes_judge": False, "reason":"empty canonical_text"}
    rid = (roleset_id or "").lower()
    if "boolean" in rid:
        return judge_boolean(task_prompt, envelope)
    if "headline" in rid or "xsum" in rid:
        return judge_headline(task_prompt, envelope)
    if "paraphrase" in rid:
        return judge_paraphrase(task_prompt, envelope)
    if "mcq" in rid:
        return judge_mcq_arc(task_prompt, envelope)
    if "wino" in rid:
        return judge_winogrande(task_prompt, envelope)
    # pattern heuristics
    if "positives:" in task_prompt and "negatives:" in task_prompt:
        return judge_regex(task_prompt, envelope)
    if "select" in ct.lower() and " from " in ct.lower():
        return judge_sql(task_prompt, envelope)
    if ct.strip().upper() in {"TRUE","FALSE"}:
        return judge_boolean(task_prompt, envelope)
    # fallback: JSON validation if looks like JSON
    if ct.strip().startswith("{") or ct.strip().startswith("["):
        return judge_json(task_prompt, envelope)
    # fallback: math attempt
    return judge_math(task_prompt, envelope)
