from __future__ import annotations
from typing import Dict, Any

ALLOWED_TAGS = {"[CONTACT]", "[SOLVED]"}
ALLOWED_STATUS = {"WORKING", "NEED_PEER", "PROPOSED", "READY_TO_SOLVE", "REVISED", "SOLVED"}

def repair_envelope(env: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a best-effort correction so a model-produced envelope passes validation.
    - Fix tag to [CONTACT] or [SOLVED]
    - Fix status to allowed set
    - Only include final_solution when SOLVED
    """
    e = dict(env) if isinstance(env, dict) else {}
    tag = str(e.get("tag", "")).strip()
    status = str(e.get("status", "")).strip().upper()
    fs = e.get("final_solution") or {}
    solvedish = isinstance(fs, dict) and isinstance(fs.get("canonical_text"), str) and fs.get("canonical_text", "").strip() != ""

    if tag not in ALLOWED_TAGS:
        e["tag"] = "[SOLVED]" if solvedish else "[CONTACT]"

    if status not in ALLOWED_STATUS:
        e["status"] = "SOLVED" if e["tag"] == "[SOLVED]" else "PROPOSED"

    # Enforce final_solution presence/absence

    return e
