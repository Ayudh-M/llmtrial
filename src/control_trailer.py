from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .sanitize import ALLOWED_STATUS, TAG_PATTERN, repair_envelope
from .utils import ALLOWED_PERFORMATIVES


CTRL_BLOCK_PATTERN = re.compile(r"<<<CTRL\s*(\{.*?\})\s*CTRL>>>", re.DOTALL)


def extract_control_trailer(text: str) -> Optional[Tuple[str, str]]:
    """Return the body (without trailing markers) and the JSON trailer."""

    if not isinstance(text, str):
        return None

    matches = list(CTRL_BLOCK_PATTERN.finditer(text))
    if not matches:
        return None

    last = matches[-1]
    body = text[: last.start()].rstrip()
    json_block = last.group(1)
    return body, json_block


def _validate_string(value: Any, field: str) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return f"Missing or empty '{field}'."
    return None


def validate_control_payload(payload: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []

    tag_error = _validate_string(payload.get("tag"), "tag")
    if tag_error:
        errors.append(tag_error)
    else:
        if not TAG_PATTERN.fullmatch(str(payload["tag"]).strip().upper()):
            errors.append("Tag must be an uppercase token in brackets, e.g. [PLAN].")

    status_error = _validate_string(payload.get("status"), "status")
    if status_error:
        errors.append(status_error)
    else:
        status = str(payload["status"]).strip().upper()
        if status not in ALLOWED_STATUS:
            allowed = ", ".join(sorted(ALLOWED_STATUS))
            errors.append(f"Invalid status '{status}'. Allowed: {allowed}.")

    intent_error = _validate_string(payload.get("intent"), "intent")
    if intent_error:
        errors.append(intent_error)
    else:
        intent = str(payload["intent"]).strip().upper()
        if intent not in ALLOWED_PERFORMATIVES:
            allowed = ", ".join(ALLOWED_PERFORMATIVES)
            errors.append(f"Unknown intent '{intent}'. Allowed intents: {allowed}.")

    if "final_solution" in payload and payload.get("final_solution") is not None:
        final_solution = payload["final_solution"]
        if not isinstance(final_solution, Mapping):
            errors.append("'final_solution' must be an object with canonical_text.")
        else:
            canonical = final_solution.get("canonical_text")
            if not isinstance(canonical, str) or not canonical.strip():
                errors.append("'final_solution.canonical_text' must be a non-empty string.")

    if "errors" in payload:
        errors_field = payload["errors"]
        if not isinstance(errors_field, Sequence) or isinstance(errors_field, (str, bytes)):
            errors.append("'errors' must be a list of strings when provided.")
        else:
            for idx, item in enumerate(errors_field):
                if not isinstance(item, str):
                    errors.append(f"errors[{idx}] must be a string.")
                    break

    return errors


def _clean_intent(intent: Any) -> Optional[str]:
    if not isinstance(intent, str):
        return None
    stripped = intent.strip().upper()
    if not stripped:
        return None
    if stripped not in ALLOWED_PERFORMATIVES:
        return None
    return stripped


def envelope_from_payload(payload: Mapping[str, Any], *, body: str = "") -> Dict[str, Any]:
    envelope: Dict[str, Any] = {
        "tag": str(payload.get("tag", "")),
        "status": str(payload.get("status", "")),
    }

    content: Dict[str, Any] = {}

    intent = _clean_intent(payload.get("intent"))
    if intent:
        content["intent"] = intent

    for field in ("acl", "verdict", "notes", "message", "next_action"):
        if field in payload and payload[field] is not None:
            content[field] = payload[field]

    if "errors" in payload and isinstance(payload["errors"], Sequence) and not isinstance(payload["errors"], (str, bytes)):
        content["errors"] = list(payload["errors"])  # type: ignore[arg-type]

    if body:
        content.setdefault("body", body)

    if content:
        envelope["content"] = content

    final_solution = payload.get("final_solution")
    if isinstance(final_solution, Mapping):
        canonical = final_solution.get("canonical_text")
        if isinstance(canonical, str) and canonical.strip():
            envelope["final_solution"] = {"canonical_text": canonical.strip()}

    return repair_envelope(envelope)


CONTROL_TRAILER_GUIDE = (
    "You may write the main body of your reply in any format suitable for your strategy. "
    "However, you MUST terminate every message with a single control trailer so the "
    "coordinator can track state and consensus. The trailer format is:\n"
    "<<<CTRL{\"tag\":\"[PLAN]\",\"status\":\"PROPOSED\",\"intent\":\"PLAN\"}CTRL>>>\n"
    "Rules:\n"
    "1. Keep the trailer on the final line of the message.\n"
    "2. Fill in tag, status, and intent with the appropriate uppercase values.\n"
    "3. When proposing or accepting a solution, include final_solution.canonical_text in the trailer, e.g.\n"
    "   \"final_solution\":{\"canonical_text\":\"ANSWER: 60 km/h\"}.\n"
    "4. When accepting a partner's solution, use tag='[SOLVED]', status='SOLVED', intent='SOLVED', and set content.verdict='ACCEPT' in your body or trailer metadata.\n"
    "Messages without a valid trailer will be rejected and you will be asked to retry."
)


__all__ = [
    "CONTROL_TRAILER_GUIDE",
    "CTRL_BLOCK_PATTERN",
    "envelope_from_payload",
    "extract_control_trailer",
    "validate_control_payload",
]
