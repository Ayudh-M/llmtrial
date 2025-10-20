from __future__ import annotations

from typing import Any, Dict

ALLOWED_TAGS = {"[CONTACT]", "[SOLVED]"}
ALLOWED_STATUS = {
    "WORKING",
    "NEED_PEER",
    "PROPOSED",
    "READY_TO_SOLVE",
    "REVISED",
    "SOLVED",
}


def repair_envelope(env: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort repair for envelopes produced by language models.

    The helper ensures that the ``tag`` and ``status`` fields are present and consistent with
    the final solution payload. It avoids mutating the original mapping by operating on a
    shallow copy.
    """

    fixed = dict(env) if isinstance(env, dict) else {}

    tag = str(fixed.get("tag", "")).strip().upper()
    status = str(fixed.get("status", "")).strip().upper()
    final_solution = fixed.get("final_solution")
    has_final = isinstance(final_solution, dict) and bool(
        str(final_solution.get("canonical_text", "")).strip()
    )

    if tag not in ALLOWED_TAGS:
        fixed["tag"] = "[SOLVED]" if has_final else "[CONTACT]"
    else:
        fixed["tag"] = tag

    if status not in ALLOWED_STATUS:
        fixed["status"] = "SOLVED" if fixed["tag"] == "[SOLVED]" else "PROPOSED"
    else:
        fixed["status"] = status

    if fixed["tag"] != "[SOLVED]":
        fixed.pop("final_solution", None)
    elif has_final:
        fixed["final_solution"] = {
            "canonical_text": str(final_solution.get("canonical_text", "")).strip(),
        }

    return fixed


__all__ = ["repair_envelope", "ALLOWED_TAGS", "ALLOWED_STATUS"]
