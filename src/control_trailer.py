from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional

# Human guidance you embed in prompts (safe to keep short in prod)
CONTROL_TRAILER_GUIDE = (
    "After your free-form message body, end with a single line control trailer:\n"
    "<<<CTRL{ ...valid compact JSON payload... }CTRL>>>\n"
    "The trailer must be LAST (no text after it)."
)

# Public constants expected by model_loader/agents
CTRL_PREFIX = "<<<CTRL"
CTRL_SUFFIX = "CTRL>>>"

# End-anchored matcher implemented via manual brace scanning.
def extract_control_trailer(text: str) -> Dict[str, Any]:
    """
    Returns: {
      'ok': bool,
      'body': str,                     # text before the trailer
      'payload': Optional[dict],       # parsed JSON or None
      'error': Optional[str],
      'offsets': {'json_start': int, 'json_end': int, 'suffix_at_end': bool}
    }
    """
    out: Dict[str, Any] = {
        "ok": False,
        "body": text,
        "payload": None,
        "error": "NOT_FOUND",
        "offsets": {"json_start": -1, "json_end": -1, "suffix_at_end": False},
    }
    prefix_pos = text.rfind(CTRL_PREFIX)
    if prefix_pos != -1:
        out["body"] = text[:prefix_pos]
    cursor = prefix_pos + len(CTRL_PREFIX)
    length = len(text)
    while cursor < length and text[cursor].isspace():
        cursor += 1

    if cursor >= length or text[cursor] != "{":
        out["error"] = "ERR_TRAILER_UNCLOSED"
        out["offsets"].update(
            {
                "json_start": cursor if cursor < length else -1,
                "json_end": length,
                "suffix_at_end": False,
            }
        )
        return out

    json_start = cursor
    depth = 0
    in_string = False
    escape = False
    json_end = -1
    idx = cursor

    while idx < length:
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == '{':
                depth += 1
            elif char == '}':
                if depth:
                    depth -= 1
                    if depth == 0:
                        json_end = idx + 1
                        break
                else:
                    break
        idx += 1

    if json_end == -1:
        out["error"] = "ERR_TRAILER_UNCLOSED"
        out["offsets"].update(
            {
                "json_start": json_start,
                "json_end": length,
                "suffix_at_end": False,
            }
        )
        return out

    suffix_start = json_end
    if not text.startswith(CTRL_SUFFIX, suffix_start) or suffix_start + len(CTRL_SUFFIX) != length:
        out["error"] = "SUFFIX_NOT_AT_END"
        out["offsets"].update(
            {
                "json_start": json_start,
                "json_end": json_end,
                "suffix_at_end": False,
            }
        )
        return out

    json_text = text[json_start:json_end]
    try:
        payload = json.loads(json_text)
    except Exception as exc:  # pragma: no cover - propagate message detail
        out["error"] = f"JSON_DECODE_ERROR: {exc}"
        out["offsets"].update(
            {
                "json_start": json_start,
                "json_end": json_end,
                "suffix_at_end": True,
            }
        )
        return out

    out.update(
        {
            "ok": True,
            "body": text[:prefix_pos],
            "payload": payload,
            "error": None,
            "offsets": {"json_start": json_start, "json_end": json_end, "suffix_at_end": True},
        }
    )
    return out


# Very lightweight schema checks so the controller can decide what to do next.
ALLOWED_TAGS = {"[PLAN]", "[SOLVER]", "[CONTACT]", "[SOLVED]"}
ALLOWED_STATUS = {"WORKING", "NEED_PEER", "PROPOSED", "READY_TO_SOLVE", "REVISED", "SOLVED"}


def _normalize_numeric(text: str) -> Optional[str]:
    try:
        value = Decimal(text)
    except (InvalidOperation, TypeError):
        return None
    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def _norm_canonical(text: Optional[str]) -> str:
    if text is None:
        return ""
    collapsed = " ".join(str(text).strip().split())
    if not collapsed:
        return ""

    # Normalize plain numbers and ANSWER: <number> style payloads.
    normalized_number = _normalize_numeric(collapsed)
    if normalized_number is not None:
        return normalized_number

    prefix, sep, remainder = collapsed.partition(":")
    if sep:
        numeric = _normalize_numeric(remainder.strip())
        if numeric is not None:
            return f"{prefix.strip()}: {numeric}".strip()

    return collapsed


def validate_control_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    errs = []
    tag = payload.get("tag")
    status = payload.get("status")
    if tag not in ALLOWED_TAGS:
        errs.append(f"tag_invalid:{tag}")
    if status not in ALLOWED_STATUS:
        errs.append(f"status_invalid:{status}")
    # Optional ACL & final_solution checks (don’t block if absent)
    final = payload.get("final_solution") or {}
    if isinstance(final, dict) and "canonical_text" in final:
        final["canonical_text"] = _norm_canonical(final.get("canonical_text"))
    return {"ok": not errs, "errors": errs, "payload": payload}


def envelope_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    # normalize shape used by controller/logger
    env = {
        "tag": payload.get("tag"),
        "status": payload.get("status"),
        "content": payload.get("content") or {},
        "final_solution": payload.get("final_solution") or {},
        "telemetry": payload.get("telemetry") or {},
    }
    if "canonical_text" in env["final_solution"]:
        env["final_solution"]["canonical_text"] = _norm_canonical(
            env["final_solution"]["canonical_text"]
        )
    return env


def normalize_canonical_text(text: Optional[str]) -> str:
    return _norm_canonical(text)


def normalise_canonical_text(text: Optional[str]) -> str:
    return _norm_canonical(text)


__all__ = [
    "CONTROL_TRAILER_GUIDE",
    "CTRL_PREFIX",
    "CTRL_SUFFIX",
    "extract_control_trailer",
    "validate_control_payload",
    "envelope_from_payload",
    "normalize_canonical_text",
    "normalise_canonical_text",
]
