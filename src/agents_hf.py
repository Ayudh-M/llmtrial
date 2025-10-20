from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .control_trailer import (
    CONTROL_TRAILER_GUIDE,
    envelope_from_payload,
    extract_control_trailer,
    validate_control_payload,
)
from .model_loader import generate_json_only
from .pseudocode import augment_system_prompt
from .sanitize import ALLOWED_STATUS, repair_envelope
from .strategies import Strategy
from .utils import ALLOWED_PERFORMATIVES, ACLParseError, parse_acl_message


_PERFORMATIVE_LIST = ", ".join(ALLOWED_PERFORMATIVES)

TRAILER_REMINDER = (
    "Control trailer reminder:\n"
    "- Place a single line trailer at the end of the message in the form <<<CTRL{...}CTRL>>>.\n"
    "- Include tag, status, and intent (allowed intents: "
    f"{_PERFORMATIVE_LIST}).\n"
    "- When proposing or accepting a solution, add final_solution.canonical_text.\n"
    "Messages without a valid trailer will be rejected."
)


def _iter_json_objects(text: str) -> List[str]:
    results: List[str] = []
    if not isinstance(text, str):
        return results

    depth = 0
    start: Optional[int] = None
    in_string = False
    escape = False

    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == '{':
            if depth == 0:
                start = index
            depth += 1
        elif char == '}':
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    results.append(text[start : index + 1])
                    start = None

    return results


def _extract_last_json(text: str) -> Optional[str]:
    blocks = _iter_json_objects(text)
    if not blocks:
        return None
    return blocks[-1]


def _validate_envelope_candidate(candidate: Any) -> List[str]:
    if not isinstance(candidate, Mapping):
        return ["Response must be a JSON object with the expected fields."]

    errors: List[str] = []

    tag = candidate.get("tag")
    if not isinstance(tag, str) or not tag.strip():
        errors.append("Missing or empty 'tag' field.")

    status = candidate.get("status")
    if not isinstance(status, str) or status.strip().upper() not in ALLOWED_STATUS:
        allowed = ", ".join(sorted(ALLOWED_STATUS))
        errors.append(f"Invalid 'status'. Expected one of: {allowed}.")

    content = candidate.get("content")
    if not isinstance(content, Mapping):
        errors.append("'content' must be an object containing an 'acl' field.")
    else:
        acl = content.get("acl")
        if not isinstance(acl, str) or not acl.strip():
            errors.append("'content.acl' must be a non-empty string following 'INTENT: message => next_action'.")
        else:
            try:
                parse_acl_message(acl)
            except ACLParseError as exc:
                errors.append(f"Invalid ACL message: {exc}")

    if "final_solution" in candidate and candidate.get("final_solution") is not None:
        final_solution = candidate.get("final_solution")
        if not isinstance(final_solution, Mapping):
            errors.append("'final_solution' must be an object when provided.")
        else:
            canonical_text = final_solution.get("canonical_text")
            if not isinstance(canonical_text, str) or not canonical_text.strip():
                errors.append("'final_solution.canonical_text' must be a non-empty string when final_solution is supplied.")

    return errors


def _retry_instructions(errors: Sequence[str]) -> str:
    bullet_list = "\n".join(f"- {msg}" for msg in errors if msg)
    details = f"\n{bullet_list}" if bullet_list else ""
    return (
        "Your previous reply was not a valid JSON envelope the coordinator could parse."
        f"{details}\n"
        "Return a single JSON object with tag, status, content.acl, and (if applicable) final_solution.canonical_text matching the schema."
    )


def _maybe_add_snippet(strategy: Strategy) -> Optional[str]:
    meta = strategy.metadata or {}
    snippet = meta.get("prompt_snippet")
    if not snippet:
        return None
    return str(snippet)


class HFChatAgent:
    """Wrapper that turns a HF causal LM into a controller-compatible agent."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tokenizer,
        model,
        strategy: Strategy,
    ) -> None:
        self.name = name
        self.base_system_prompt = augment_system_prompt(system_prompt)
        self.tokenizer = tokenizer
        self.model = model
        self.strategy = strategy

    # -- prompt assembly -------------------------------------------------
    def _system_prompt(self, preparation: Optional[Mapping[str, Any]]) -> str:
        prep = preparation or {}
        parts: List[str] = []
        if prep.get("system_prefix"):
            parts.append(str(prep["system_prefix"]))
        parts.append(self.base_system_prompt)

        snippet = _maybe_add_snippet(self.strategy)
        if snippet:
            parts.append(snippet)

        if not prep.get("omit_json_guide"):
            parts.append(CONTROL_TRAILER_GUIDE)

        if prep.get("system_suffix"):
            parts.append(str(prep["system_suffix"]))

        return "\n\n".join([p for p in parts if p])

    def _user_prompt(
        self,
        task: str,
        transcript: List[Dict[str, Any]],
        preparation: Optional[Mapping[str, Any]],
    ) -> str:
        prep = preparation or {}
        peer_context = "{}"
        if transcript:
            peer_context = json.dumps(transcript[-1].get("envelope", {}), ensure_ascii=False)

        parts: List[str] = []
        if prep.get("user_prefix"):
            parts.append(str(prep["user_prefix"]))

        base = (
            f"Task: {task}\n"
            f"Peer context: {peer_context}\n"
            "Respond in your preferred style, then append the control trailer exactly as instructed."
        )
        prompt_context = {"agent": self.name}
        decorated = self.strategy.decorate_prompts(base, prompt_context)
        parts.append(decorated)
        parts.append(TRAILER_REMINDER)

        if prep.get("user_suffix"):
            parts.append(str(prep["user_suffix"]))
        if prep.get("extra_user_instructions"):
            parts.append(str(prep["extra_user_instructions"]))

        return "\n\n".join([p for p in parts if p])

    def _messages(
        self,
        task: str,
        transcript: List[Dict[str, Any]],
        preparation: Optional[Mapping[str, Any]],
    ) -> List[Dict[str, str]]:
        system_prompt = self._system_prompt(preparation)
        user_prompt = self._user_prompt(task, transcript, preparation)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    # -- controller hook -------------------------------------------------
    def step(
        self,
        task: str,
        transcript: List[Dict[str, Any]],
        preparation: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], str]:
        base_messages = self._messages(task, transcript, preparation)
        decoding: Dict[str, Any] = dict(self.strategy.decoding or {})
        if preparation and preparation.get("decoding_override"):
            decoding.update(preparation["decoding_override"])  # type: ignore[arg-type]

        max_attempts = 3
        errors: List[str] = []
        raw_output = ""

        for attempt in range(max_attempts):
            convo: List[Dict[str, str]] = list(base_messages)
            if attempt and errors:
                convo.append({"role": "user", "content": _retry_instructions(errors)})

            raw_output = generate_json_only(
                self.tokenizer,
                self.model,
                convo,
                decoding=decoding,
            )

            json_payload = _extract_last_json(raw_output)
            if not json_payload:
                errors = ["Response did not contain a JSON object."]
                continue

            try:
                candidate = json.loads(json_payload)
            except json.JSONDecodeError as exc:
                errors = [
                    f"Invalid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})."
                ]
                continue

            errors = _validate_envelope_candidate(candidate)
            if errors:
                continue

            envelope = repair_envelope(candidate)
            return envelope, raw_output

        fallback: Dict[str, Any] = {
            "tag": "[CONTACT]",
            "status": "NEED_PEER",
            "content": {
                "acl": "QUESTION: Unable to parse your last reply => WAIT_FOR_PEER",
            },
        }
        if errors:
            fallback["content"]["errors"] = list(errors)

        return repair_envelope(fallback), raw_output


__all__ = ["HFChatAgent", "JSON_GUIDE"]

