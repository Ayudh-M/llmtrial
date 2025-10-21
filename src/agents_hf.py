from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .control_trailer import (
    CONTROL_TRAILER_GUIDE,
    ControlTrailerFailure,
    ControlTrailerValidation,
    envelope_from_payload,
    extract_control_trailer,
    validate_control_payload,
)
from .model_loader import GenerationResult, generate_json_only
from .pseudocode import augment_system_prompt
from .strategies import Strategy
from .utils import ALLOWED_PERFORMATIVES


_PERFORMATIVE_LIST = ", ".join(ALLOWED_PERFORMATIVES)

TRAILER_REMINDER = (
    "Control trailer reminder:\n"
    "- Keep the body concise so the trailer fits (reserve ~80 tokens).\n"
    "- Place a single line trailer at the end of the message in the form <<<CTRL{...}CTRL>>>.\n"
    "- Nothing may appear after the trailer. If you need more space, say it before the trailer.\n"
    "- Include tag, status, and intent (allowed intents: "
    f"{_PERFORMATIVE_LIST}).\n"
    "- When proposing or accepting a solution, add final_solution.canonical_text beginning with 'ANSWER: '.\n"
    "Messages without a valid trailer will be rejected."
)

BODY_PREVIEW_LIMIT = 600


def _truncate(text: str, limit: int = BODY_PREVIEW_LIMIT) -> str:
    if not isinstance(text, str):
        return ""
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    return text[: limit - 1] + "…"


REASON_HINTS: Dict[str, str] = {
    "missing_trailer": "You must finish with <<<CTRL{...}CTRL>>>.",
    "not_at_end": "Place the trailer at the end with no extra text after CTRL>>>.",
    "malformed_json": "Ensure the trailer JSON is valid and balanced.",
    "missing_canonical": "Statuses READY_TO_SOLVE/PROPOSED/SOLVED require final_solution.canonical_text beginning with 'ANSWER: '.",
    "illegal_transition": "Check tag/status/intent values against the allowed list and follow the proposer/acceptor handshake rules.",
}


def _retry_instructions(errors: List[str], reasons: List[str]) -> str:
    hints = [REASON_HINTS.get(code, "") for code in reasons]
    combined = [msg for msg in (*errors, *hints) if msg]
    bullet_list = "\n".join(f"- {msg}" for msg in combined)
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

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        tokenizer = getattr(self, "tokenizer", None)
        encode = getattr(tokenizer, "encode", None) if tokenizer is not None else None
        if encode is not None:
            try:
                tokens = encode(text, add_special_tokens=False)  # type: ignore[arg-type]
            except TypeError:
                tokens = encode(text)  # type: ignore[arg-type]
            except Exception:
                tokens = None
            if isinstance(tokens, (list, tuple)):
                return len(tokens)
            if hasattr(tokens, "input_ids"):
                candidate = getattr(tokens, "input_ids")
                if hasattr(candidate, "__len__"):
                    return len(candidate)  # type: ignore[arg-type]
            if hasattr(tokens, "__len__"):
                return len(tokens)  # type: ignore[arg-type]
        stripped = text.strip()
        if not stripped:
            return 0
        return len(stripped.split())

    # -- prompt assembly -------------------------------------------------
    def _adjust_decoding(self, decoding: Dict[str, Any]) -> None:
        metadata = self.strategy.metadata or {}
        body_style = str(metadata.get("body_style", "json")).strip().lower()
        if body_style in {"json", "dsl", "kqml"}:
            decoding["do_sample"] = False
            decoding["temperature"] = 0.0
        else:
            if "do_sample" not in decoding:
                decoding["do_sample"] = True
            if decoding.get("do_sample"):
                decoding.setdefault("temperature", float(decoding.get("temperature") or 0.5))
            else:
                decoding.setdefault("temperature", 0.0)

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
        self._adjust_decoding(decoding)

        max_attempts = 3
        error_messages: List[str] = []
        failure_reasons: List[str] = []
        last_result: Optional[GenerationResult] = None
        raw_output = ""

        for attempt in range(max_attempts):
            if attempt and (error_messages or failure_reasons):
                retry_content = base_user_content
                retry_hint = _retry_instructions(error_messages, failure_reasons)
                if retry_content:
                    retry_content = f"{retry_content}\n\n{retry_hint}"
                else:
                    retry_content = retry_hint
                convo = [
                    dict(system_message),
                    {
                        "role": "user",
                        "content": retry_content,
                    },
                ]
            else:
                convo = [dict(system_message), dict(user_message)]

            result = generate_json_only(
                self.tokenizer,
                self.model,
                convo,
                decoding=decoding,
            )
            raw_output = result.text
            last_result = result

            extraction, failure = extract_control_trailer(raw_output)
            if failure:
                error_messages = [failure.message or "Missing <<<CTRL{...}CTRL>>> control trailer at end of message."]
                if failure.reason and failure.reason not in failure_reasons:
                    failure_reasons.append(failure.reason)
                continue

            assert extraction is not None
            body = extraction.body
            json_payload = extraction.json_block
            try:
                candidate = json.loads(json_payload)
            except json.JSONDecodeError as exc:
                error_messages = [
                    f"Invalid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})."
                ]
                if "malformed_json" not in failure_reasons:
                    failure_reasons.append("malformed_json")
                continue

            if not isinstance(candidate, Mapping):
                error_messages = ["Control trailer must contain a JSON object with tag, status, and intent."]
                if "illegal_transition" not in failure_reasons:
                    failure_reasons.append("illegal_transition")
                continue

            validation: ControlTrailerValidation = validate_control_payload(candidate)
            if not validation.ok:
                error_messages = list(validation.errors)
                if validation.reason and validation.reason not in failure_reasons:
                    failure_reasons.append(validation.reason)
                continue

            stop_reason = None
            stopped_on_ctrl = False
            if last_result is not None:
                stop_reason = last_result.stop_reason
                stopped_on_ctrl = last_result.stop_reason == "ctrl" or last_result.trailer_triggered
            trailer_text = raw_output[extraction.trailer_start : extraction.trailer_end]
            tokens_trailer = self._count_tokens(trailer_text)
            total_tokens = last_result.new_tokens if last_result else 0
            tokens_body = max(total_tokens - tokens_trailer, 0)
            tokens_reserved = last_result.reserved_tokens if last_result else 0
            tokens_overflow = max(tokens_trailer - tokens_reserved, 0)
            body_budget = last_result.body_budget if last_result else 0
            body_overflow = max(tokens_body - body_budget, 0)
            has_tail = bool(raw_output[extraction.trailer_end :].strip())
            telemetry = {
                "retry_count": attempt,
                "first_error": failure_reasons[0] if failure_reasons else None,
                "body_len": len(body),
                "trailer_len": len(json_payload),
                "stop_reason": stop_reason,
                "stopped_on": stop_reason,
                "stopped_on_ctrl": stopped_on_ctrl,
                "new_tokens": last_result.new_tokens if last_result else None,
                "max_new_tokens": last_result.max_new_tokens if last_result else None,
                "input_tokens": last_result.input_tokens if last_result else None,
                "body_budget": last_result.body_budget if last_result else None,
                "reserved_tokens": last_result.reserved_tokens if last_result else None,
                "trailer_triggered": last_result.trailer_triggered if last_result else None,
                "trailer_start": extraction.trailer_start,
                "trailer_end": extraction.trailer_end,
                "has_tail": has_tail,
                "tokens_reserved": tokens_reserved,
                "tokens_used_total": total_tokens,
                "tokens_used_body": tokens_body,
                "tokens_used_trailer": tokens_trailer,
                "tokens_overflow": tokens_overflow,
                "tokens_body_overflow": body_overflow,
                "tokens_body_budget": body_budget,
            }
            control_meta = {
                "first_error": failure_reasons[0] if failure_reasons else None,
                "errors": list(failure_reasons),
                "body_preview": _truncate(body),
                "trailer": json_payload,
                "strategy_id": self.strategy.id,
                "strategy_name": self.strategy.name,
                "strategy_body_style": (self.strategy.metadata or {}).get("body_style"),
                "source": "trailer",
                "telemetry": {
                    k: v
                    for k, v in telemetry.items()
                if v is not None
                or isinstance(v, bool)
                or k in {"retry_count", "body_len", "trailer_len", "trailer_start", "trailer_end"}
            },
            }
            final_solution = candidate.get("final_solution")
            if isinstance(final_solution, Mapping):
                canonical_raw = final_solution.get("canonical_text")
                if isinstance(canonical_raw, str):
                    control_meta.setdefault("raw_canonical", canonical_raw)

            envelope = envelope_from_payload(
                candidate,
                body=body,
                trailer=json_payload,
                meta=control_meta,
            )
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
        if error_messages:
            fallback_errors = list(error_messages)
            fallback["content"]["errors"] = fallback_errors
            payload["errors"] = fallback_errors

        telemetry = {
            "retry_count": max_attempts - 1,
            "first_error": failure_reasons[0] if failure_reasons else None,
            "body_len": len(base_user_content),
            "stop_reason": last_result.stop_reason if last_result else None,
            "stopped_on": last_result.stop_reason if last_result else None,
            "stopped_on_ctrl": bool(last_result.stop_reason == "ctrl") if last_result else False,
            "new_tokens": last_result.new_tokens if last_result else None,
            "max_new_tokens": last_result.max_new_tokens if last_result else None,
            "input_tokens": last_result.input_tokens if last_result else None,
            "body_budget": last_result.body_budget if last_result else None,
            "reserved_tokens": last_result.reserved_tokens if last_result else None,
            "trailer_triggered": last_result.trailer_triggered if last_result else None,
            "tokens_reserved": last_result.reserved_tokens if last_result else None,
            "tokens_used_total": last_result.new_tokens if last_result else 0,
            "tokens_used_body": last_result.new_tokens if last_result else 0,
            "tokens_used_trailer": 0,
            "tokens_overflow": 0,
            "tokens_body_overflow": 0,
            "tokens_body_budget": last_result.body_budget if last_result else None,
            "has_tail": False,
        }
        control_meta = {
            "first_error": failure_reasons[0] if failure_reasons else None,
            "errors": list(failure_reasons),
            "body_preview": _truncate(base_user_content),
            "strategy_id": self.strategy.id,
            "strategy_name": self.strategy.name,
            "strategy_body_style": (self.strategy.metadata or {}).get("body_style"),
            "source": "fallback",
            "telemetry": {
                k: v
                for k, v in telemetry.items()
                if v is not None or isinstance(v, bool) or k in {"retry_count", "body_len"}
            },
        }

        return (
            envelope_from_payload(payload, body=base_user_content, meta={**control_meta, "source": "fallback"}),
            raw_output,
        )


__all__ = ["HFChatAgent", "CONTROL_TRAILER_GUIDE"]

