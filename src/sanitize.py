from __future__ import annotations

import re
from typing import Any, Dict, Optional

TAG_PATTERN = re.compile(r"\[[A-Z0-9_:-]+\]")

NON_SOLVED_FINALS = {
    "PLAN_READY",
    "PLAN_NEEDS_WORK",
    "READY_FOR_TESTS",
    "NEEDS_REVISION",
    "READY",
    "BLOCKED",
    "REVISION_DONE",
    "REVISE",
    "APPROVED",
    "ALLOW",
    "DENY",
    "ESCALATE",
    "UNSURE",
    "SUMMARY_WITH_CITATIONS",
    "NEEDS_MORE",
    "DRAFT_B_V1",
    "JOINT_FINAL",
    "A",
    "B",
    "TIE",
}
ALLOWED_STATUS = {
    "WORKING",
    "NEED_PEER",
    "PROPOSED",
    "READY_TO_SOLVE",
    "REVISED",
    "SOLVED",
}


def _tag_hint(raw_tag: str) -> Optional[bool]:
    """Return True if the tag hints at SOLVED, False if CONTACT, else None."""

    tag = raw_tag.strip().upper()
    if not tag.startswith("["):
        return None
    if tag.startswith("[SOLVED"):
        return True
    if tag.startswith("[CONTACT"):
        return False
    return None


def repair_envelope(env: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort repair for envelopes produced by language models.

    The helper ensures that the ``tag`` and ``status`` fields are present and consistent with
    the final solution payload. It avoids mutating the original mapping by operating on a
    shallow copy.
    """

    fixed = dict(env) if isinstance(env, dict) else {}

    raw_tag = str(fixed.get("tag", ""))
    status = str(fixed.get("status", "")).strip().upper()
    final_solution = fixed.get("final_solution")
    canonical_text = ""
    if isinstance(final_solution, dict):
        raw_value = final_solution.get("canonical_text", "")
        canonical_text = str(raw_value) if raw_value is not None else ""

    normalised_tag = raw_tag.strip().upper()
    canonical_trimmed = canonical_text.strip()
    canonical_upper = canonical_trimmed.upper() if canonical_trimmed else ""
    stage_like = canonical_upper in NON_SOLVED_FINALS
    tag_hint = _tag_hint(raw_tag)

    if tag_hint is True and stage_like:
        target_tag = "[CONTACT]"
    elif TAG_PATTERN.fullmatch(normalised_tag):
        target_tag = normalised_tag
    elif canonical_trimmed and not stage_like:
        target_tag = "[SOLVED]"
    else:
        target_tag = "[CONTACT]"

    fixed["tag"] = target_tag

    downgraded_from_solved = tag_hint is True and target_tag != "[SOLVED]"

    if target_tag == "[SOLVED]":
        fixed["status"] = "SOLVED"
    elif downgraded_from_solved:
        fixed["status"] = "PROPOSED"
    elif status in ALLOWED_STATUS:
        fixed["status"] = status
    else:
        fixed["status"] = "PROPOSED"

    if isinstance(final_solution, dict) and canonical_trimmed:
        fixed["final_solution"] = {"canonical_text": canonical_trimmed}
    else:
        fixed.pop("final_solution", None)

    return fixed


__all__ = [
    "repair_envelope",
    "ALLOWED_STATUS",
    "TAG_PATTERN",
    "NON_SOLVED_FINALS",
]
