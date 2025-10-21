from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .sanitize import ALLOWED_STATUS, TAG_PATTERN, repair_envelope
from .utils import ALLOWED_PERFORMATIVES


CTRL_PREFIX = "<<<CTRL"
CTRL_SUFFIX = "CTRL>>>"


@dataclass
class ControlTrailerExtraction:
    body: str
    json_block: str
    trailer_start: int
    trailer_end: int


@dataclass
class ControlTrailerFailure:
    reason: str
    message: str = ""


@dataclass
class ControlTrailerValidation:
    errors: List[str]
    reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        return not self.errors


def _balanced_json(payload: str) -> bool:
    depth = 0
    in_string = False
    escape = False
    for char in payload:
        if in_string:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return False

    return depth == 0 and not in_string


def extract_control_trailer(text: str) -> Tuple[Optional[ControlTrailerExtraction], Optional[ControlTrailerFailure]]:
    """Return the terminal control trailer or a structured failure."""

    if not isinstance(text, str):
        return None, ControlTrailerFailure("missing_trailer", "Response must be textual.")

    end_index = text.rfind(CTRL_SUFFIX)
    if end_index == -1:
        return None, ControlTrailerFailure("missing_trailer", "No <<<CTRL{...}CTRL>>> trailer found.")

    suffix_end = end_index + len(CTRL_SUFFIX)
    tail = text[suffix_end:]
    if tail.strip():
        return None, ControlTrailerFailure("not_at_end", "Control trailer must terminate the message with nothing after CTRL>>>")

    prefix_index = text.rfind(CTRL_PREFIX, 0, end_index)
    if prefix_index == -1:
        return None, ControlTrailerFailure("missing_trailer", "Could not locate the opening <<<CTRL marker for the trailer.")

    raw_payload = text[prefix_index + len(CTRL_PREFIX) : end_index]
    json_block = raw_payload.strip()
    if not json_block:
        return None, ControlTrailerFailure("malformed_json", "Trailer payload between <<<CTRL and CTRL>>> is empty.")

    if not json_block.startswith("{") or not json_block.endswith("}"):
        return None, ControlTrailerFailure("malformed_json", "Trailer payload must be a JSON object enclosed in braces.")

    if not _balanced_json(json_block):
        return None, ControlTrailerFailure("malformed_json", "Trailer JSON braces or strings are unbalanced.")

    body = text[:prefix_index].rstrip()
    extraction = ControlTrailerExtraction(
        body=body,
        json_block=json_block,
        trailer_start=prefix_index,
        trailer_end=suffix_end,
    )
    return extraction, None


def _validate_string(value: Any, field: str) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return f"Missing or empty '{field}'."
    return None


def normalise_canonical_text(value: str) -> str:
    text = " ".join(str(value).strip().split())
    if not text:
        return text

    upper = text.upper()
    if not upper.startswith("ANSWER:"):
        return text

    prefix, remainder = text.split(":", 1)
    remainder = " ".join(remainder.strip().split())

    numeric = remainder
    try:
        numeric_candidate = remainder.replace(",", "")
        decimal = Decimal(numeric_candidate)
        normalized = format(decimal.normalize(), "f")
        if "." in normalized:
            normalized = normalized.rstrip("0").rstrip(".") or "0"
        numeric = normalized
    except (InvalidOperation, ValueError):
        numeric = remainder

    final_remainder = numeric if numeric else remainder
    return f"ANSWER: {final_remainder}".strip()


def validate_control_payload(payload: Mapping[str, Any]) -> ControlTrailerValidation:
    errors: List[str] = []
    reason: Optional[str] = None

    tag_error = _validate_string(payload.get("tag"), "tag")
    tag_value = str(payload.get("tag", "")) if not tag_error else ""
    if tag_error:
        errors.append(tag_error)
        reason = reason or "illegal_transition"
    else:
        if not TAG_PATTERN.fullmatch(tag_value.strip().upper()):
            errors.append("Tag must be an uppercase token in brackets, e.g. [PLAN].")
            reason = reason or "illegal_transition"

    status_error = _validate_string(payload.get("status"), "status")
    status_value = str(payload.get("status", "")) if not status_error else ""
    if status_error:
        errors.append(status_error)
        reason = reason or "illegal_transition"
    else:
        status = status_value.strip().upper()
        if status not in ALLOWED_STATUS:
            allowed = ", ".join(sorted(ALLOWED_STATUS))
            errors.append(f"Invalid status '{status}'. Allowed: {allowed}.")
            reason = reason or "illegal_transition"
        else:
            status_value = status

    intent_error = _validate_string(payload.get("intent"), "intent")
    if intent_error:
        errors.append(intent_error)
        reason = reason or "illegal_transition"
    else:
        intent = str(payload["intent"]).strip().upper()
        if intent not in ALLOWED_PERFORMATIVES:
            allowed = ", ".join(ALLOWED_PERFORMATIVES)
            errors.append(f"Unknown intent '{intent}'. Allowed intents: {allowed}.")
            reason = reason or "illegal_transition"

    final_solution = payload.get("final_solution")
    requires_final = status_value in {"READY_TO_SOLVE", "SOLVED"} or tag_value.strip().upper() == "[SOLVED]"
    if payload.get("final_solution") is not None and status_value not in {"PROPOSED", "READY_TO_SOLVE", "SOLVED"}:
        errors.append("final_solution is only allowed when status is PROPOSED, READY_TO_SOLVE, or SOLVED.")
        reason = reason or "illegal_transition"

    canonical_text: Optional[str] = None
    if final_solution is not None:
        if not isinstance(final_solution, Mapping):
            errors.append("'final_solution' must be an object with canonical_text.")
            reason = reason or "illegal_transition"
        else:
            canonical = final_solution.get("canonical_text")
            if not isinstance(canonical, str) or not canonical.strip():
                errors.append("'final_solution.canonical_text' must be a non-empty string.")
                reason = reason or "missing_canonical"
            else:
                canonical_text = canonical
                normalised = normalise_canonical_text(canonical)
                if not normalised.upper().startswith("ANSWER:"):
                    errors.append("final_solution.canonical_text must start with 'ANSWER: '.")
                    reason = reason or "missing_canonical"
                else:
                    canonical_text = normalised
    if requires_final and canonical_text is None:
        errors.append("Status requires final_solution.canonical_text with an ANSWER: prefix.")
        reason = reason or "missing_canonical"

    if "errors" in payload:
        errors_field = payload["errors"]
        if not isinstance(errors_field, Sequence) or isinstance(errors_field, (str, bytes)):
            errors.append("'errors' must be a list of strings when provided.")
            reason = reason or "illegal_transition"
        else:
            for idx, item in enumerate(errors_field):
                if not isinstance(item, str):
                    errors.append(f"errors[{idx}] must be a string.")
                    reason = reason or "illegal_transition"
                    break

    return ControlTrailerValidation(errors=errors, reason=reason)


def _clean_intent(intent: Any) -> Optional[str]:
    if not isinstance(intent, str):
        return None
    stripped = intent.strip().upper()
    if not stripped:
        return None
    if stripped not in ALLOWED_PERFORMATIVES:
        return None
    return stripped


def envelope_from_payload(
    payload: Mapping[str, Any], *, body: str = "", trailer: Optional[str] = None, meta: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
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

    control_meta: Dict[str, Any] = {}
    if body:
        control_meta["body"] = body
    if trailer:
        control_meta["trailer"] = trailer
    if meta:
        control_meta.update({k: v for k, v in meta.items() if v is not None})
    if control_meta:
        content.setdefault("control", {}).update(control_meta)

    if body and "body" not in content:
        content.setdefault("body", body)

    if content:
        envelope["content"] = content

    final_solution = payload.get("final_solution")
    if isinstance(final_solution, Mapping):
        canonical = final_solution.get("canonical_text")
        if isinstance(canonical, str) and canonical.strip():
            normalised = normalise_canonical_text(canonical)
            envelope["final_solution"] = {"canonical_text": normalised}
            if control_meta is not None:
                control_meta.setdefault("raw_canonical", canonical)
                control_meta.setdefault("normalised_canonical", normalised)

    if control_meta:
        envelope.setdefault("content", {}).setdefault("control", {}).update(control_meta)

    return repair_envelope(envelope)


CONTROL_TRAILER_GUIDE = (
    "You may write the main body of your reply in any format suitable for your strategy. "
    "However, you MUST terminate every message with a single control trailer so the "
    "coordinator can track state and consensus. The trailer format is:\n"
    "<<<CTRL{\"tag\":\"[PLAN]\",\"status\":\"PROPOSED\",\"intent\":\"PLAN\"}CTRL>>>\n"
    "Rules:\n"
    "1. Keep the trailer on the final line of the message.\n"
    "2. Nothing may appear after the trailer—only trailing whitespace is allowed.\n"
    "3. Fill in tag, status, and intent with the appropriate uppercase values.\n"
    "4. When proposing or accepting a solution, include final_solution.canonical_text in the trailer with an 'ANSWER: ' prefix, e.g.\n"
    "   \"final_solution\":{\"canonical_text\":\"ANSWER: 60 km/h\"}.\n"
    "5. When accepting a partner's solution, use tag='[SOLVED]', status='SOLVED', intent='SOLVED', and set content.verdict='ACCEPT' in your body or trailer metadata.\n"
    "Messages without a valid trailer will be rejected and you will be asked to retry."
)


__all__ = [
    "CONTROL_TRAILER_GUIDE",
    "CTRL_PREFIX",
    "CTRL_SUFFIX",
    "ControlTrailerExtraction",
    "ControlTrailerFailure",
    "ControlTrailerValidation",
    "envelope_from_payload",
    "extract_control_trailer",
    "normalise_canonical_text",
    "validate_control_payload",
]
