from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .control_trailer import (
    CONTROL_TRAILER_GUIDE,
    envelope_from_payload,
    extract_control_trailer,
    validate_control_payload,
)
from .model_loader import generate_json_only
from .pseudocode import augment_system_prompt
from .strategies import Strategy
from .utils import ALLOWED_PERFORMATIVES


_PERFORMATIVE_LIST = ", ".join(ALLOWED_PERFORMATIVES)

TRAILER_REMINDER = (
    "Control trailer reminder:\n"
    "- Place a single line trailer at the end of the message in the form <<<CTRL{...}CTRL>>>.\n"
    "- Include tag, status, and intent (allowed intents: "
    f"{_PERFORMATIVE_LIST}).\n"
    "- When proposing or accepting a solution, add final_solution.canonical_text.\n"
    "Messages without a valid trailer will be rejected."
)


def _retry_instructions(errors: List[str]) -> str:
    bullet_list = "\n".join(f"- {msg}" for msg in errors if msg)
    details = f"\n{bullet_list}" if bullet_list else ""
    return (
        "Your previous reply did not include a valid control trailer."
        f"{details}\n"
        "Rewrite your message (body as needed) and finish with <<<CTRL{{...}}CTRL>>> including tag, status, intent, and any required final_solution.canonical_text."
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
        if len(base_messages) < 2:
            raise ValueError("Expected at least system and user messages for generation.")

        system_message = dict(base_messages[0])
        user_message = dict(base_messages[1])
        base_user_content = user_message.get("content", "")
        decoding: Dict[str, Any] = dict(self.strategy.decoding or {})
        if preparation and preparation.get("decoding_override"):
            decoding.update(preparation["decoding_override"])  # type: ignore[arg-type]

        max_attempts = 3
        errors: List[str] = []
        raw_output = ""

        for attempt in range(max_attempts):
            if attempt and errors:
                retry_content = base_user_content
                if retry_content:
                    retry_content = f"{retry_content}\n\n{_retry_instructions(errors)}"
                else:
                    retry_content = _retry_instructions(errors)
                convo = [
                    dict(system_message),
                    {
                        "role": "user",
                        "content": retry_content,
                    },
                ]
            else:
                convo = [dict(system_message), dict(user_message)]

            raw_output = generate_json_only(
                self.tokenizer,
                self.model,
                convo,
                decoding=decoding,
            )

            trailer = extract_control_trailer(raw_output)
            if not trailer:
                errors = ["Missing <<<CTRL{...}CTRL>>> control trailer at end of message."]
                continue

            body, json_payload = trailer
            try:
                candidate = json.loads(json_payload)
            except json.JSONDecodeError as exc:
                errors = [
                    f"Invalid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})."
                ]
                continue

            if not isinstance(candidate, Mapping):
                errors = ["Control trailer must contain a JSON object with tag, status, and intent."]
                continue

            errors = validate_control_payload(candidate)
            if errors:
                continue

            envelope = envelope_from_payload(candidate, body=body)
            return envelope, raw_output

        fallback: Dict[str, Any] = {
            "tag": "[CONTACT]",
            "status": "NEED_PEER",
            "content": {
                "intent": "QUESTION",
                "body": base_user_content,
            },
        }
        payload = {
            "tag": fallback["tag"],
            "status": fallback["status"],
            "intent": "QUESTION",
        }
        if errors:
            fallback_errors = list(errors)
            fallback["content"]["errors"] = fallback_errors
            payload["errors"] = fallback_errors

        return envelope_from_payload(payload, body=base_user_content), raw_output


__all__ = ["HFChatAgent", "CONTROL_TRAILER_GUIDE"]

