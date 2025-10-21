from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .control_trailer import (
    CONTROL_TRAILER_GUIDE,
    CTRL_PREFIX,
    CTRL_SUFFIX,
    envelope_from_payload,
    extract_control_trailer,
    validate_control_payload,
)
from .model_loader import GenerationResult, generate_json_only, generate_with_trailer
from .pseudocode import augment_system_prompt
from .sanitize import ALLOWED_STATUS, repair_envelope
from .strategies import Strategy
from .utils import ALLOWED_PERFORMATIVES, ACLParseError, parse_acl_message


_PERFORMATIVE_LIST = ", ".join(ALLOWED_PERFORMATIVES)

TRAILER_REMINDER = (
    "Control trailer reminder:\n"
    "- Place a single line trailer as the final characters of the message in the form <<<CTRL{...}CTRL>>>.\n"
    "- Include tag, status, and intent (allowed intents: "
    f"{_PERFORMATIVE_LIST}).\n"
    "- When proposing or accepting a solution, add final_solution.canonical_text.\n"
    "Messages without a valid trailer will be rejected."
)

BODY_PREVIEW_LIMIT = 600


def _truncate(text: str, limit: int = BODY_PREVIEW_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


def _retry_instructions(errors: Sequence[str]) -> str:
    bullet_list = "\n".join(f"- {msg}" for msg in errors if msg)
    details = f"\n{bullet_list}" if bullet_list else ""
    return (
        "Your previous reply was not a valid JSON envelope the coordinator could parse."
        f"{details}\n"
        "Return a single JSON object with tag, status, content.acl, and (if applicable) final_solution.canonical_text matching the schema."
    )


def _trailer_error_hint(error_code: str) -> str:
    mapping = {
        "NOT_FOUND": "You must end your reply with <<<CTRL{...}CTRL>>> containing valid JSON.",
        "SUFFIX_NOT_AT_END": "The control trailer must be the final text in the message with no trailing characters.",
    }
    if error_code.startswith("JSON_DECODE_ERROR"):
        return "The control trailer JSON was invalid; ensure it is compact valid JSON with escaped quotes."
    return mapping.get(error_code, f"Control trailer error: {error_code}.")


def _append_retry_hint(
    system_message: Mapping[str, Any],
    user_message: Mapping[str, Any],
    hint: Optional[str],
) -> List[Dict[str, str]]:
    updated_user = dict(user_message)
    if hint:
        base = str(updated_user.get("content", ""))
        updated_user["content"] = f"{base}\n\n{hint}" if base else hint
    return [dict(system_message), updated_user]


def _validate_envelope_candidate(candidate: Any) -> List[str]:
    if not isinstance(candidate, Mapping):
        return ["Response must be a JSON object with the expected fields."]

    errors: List[str] = []

    tag = candidate.get("tag")
    if not isinstance(tag, str) or not tag.strip():
        errors.append("Missing or empty 'tag' field.")

    status = candidate.get("status")
    if not isinstance(status, str) or not status.strip():
        errors.append("Missing or empty 'status' field.")
    else:
        status_upper = status.strip().upper()
        if status_upper not in ALLOWED_STATUS:
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


def _extract_last_json(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None

    depth = 0
    start: Optional[int] = None
    in_string = False
    escape = False
    last: Optional[str] = None

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
        elif char == '}' and depth:
            depth -= 1
            if depth == 0 and start is not None:
                last = text[start : index + 1]
                start = None

    return last


def _telemetry_from(
    result: GenerationResult,
    extraction: Dict[str, Any],
    failure_codes: Sequence[str],
    attempt: int,
    totals: Optional[Mapping[str, Any]] = None,
    trailer_only_retry: bool = False,
) -> Dict[str, Any]:
    offsets = extraction.get("offsets") or {}
    json_start = offsets.get("json_start", -1)
    json_end = offsets.get("json_end", -1)
    suffix_at_end = bool(offsets.get("suffix_at_end"))
    body_len = len(extraction.get("body", ""))
    trailer_len = max(json_end - json_start, 0)
    trailer_start = json_start - len(CTRL_PREFIX) if json_start >= 0 else -1
    trailer_end = json_end + len(CTRL_SUFFIX) if json_end >= 0 else -1
    totals = dict(totals or {})
    tokens_used_total = int(totals.get("tokens_used_total", result.tokens_used))
    tokens_reserved_total = int(totals.get("tokens_reserved_total", result.tokens_reserved))
    body_tokens_total = int(totals.get("body_tokens_total", result.body_tokens))
    trailer_tokens_total = int(totals.get("trailer_tokens_total", result.trailer_tokens))
    body_overflow_total = int(totals.get("tokens_body_overflow_total", result.tokens_body_overflow))
    trailer_overflow_total = int(
        totals.get("tokens_trailer_overflow_total", result.tokens_trailer_overflow)
    )
    body_budget = int(totals.get("body_budget", result.body_budget))
    trailer_budget = int(totals.get("trailer_budget", result.trailer_budget))
    closed_ctrl = suffix_at_end and not result.has_tail
    telemetry = {
        "retry_count": attempt,
        "error_log": list(failure_codes),
        "stopped_on": result.stop_reason,
        "tokens_used": result.tokens_used,
        "tokens_overflow": result.overflow_tokens,
        "has_tail": result.has_tail or not suffix_at_end,
        "trailer_offset": json_start,
        "trailer_json_end": json_end,
        "suffix_at_end": suffix_at_end,
        "body_len": body_len,
        "trailer_len": trailer_len,
        "trailer_start": trailer_start,
        "trailer_end": trailer_end,
        "stopped_on_ctrl": result.stop_reason == "suffix" and suffix_at_end,
        "tokens_used_total": tokens_used_total,
        "tokens_reserved": tokens_reserved_total,
        "tokens_body_total": body_tokens_total,
        "tokens_trailer_total": trailer_tokens_total,
        "tokens_body_overflow_total": body_overflow_total,
        "tokens_trailer_overflow_total": trailer_overflow_total,
        "tokens_body_budget": body_budget,
        "tokens_trailer_budget": trailer_budget,
        "trailer_only_retry": bool(trailer_only_retry),
        "closed_ctrl": bool(closed_ctrl),
        "first_error": failure_codes[0] if failure_codes else None,
    }
    return telemetry


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

    def _body_style(self) -> str:
        metadata = self.strategy.metadata or {}
        return str(metadata.get("body_style", "json")).strip().lower()

    # -- prompt assembly -------------------------------------------------
    def _adjust_decoding(self, decoding: Dict[str, Any]) -> None:
        body_style = self._body_style()
        if body_style in {"json", "dsl", "kqml"}:
            decoding["do_sample"] = False
            decoding.pop("temperature", None)
            decoding.pop("top_p", None)
            decoding.pop("top_k", None)
        else:
            if "do_sample" not in decoding:
                decoding["do_sample"] = True
            if decoding.get("do_sample"):
                decoding.setdefault("temperature", float(decoding.get("temperature") or 0.5))
                if "top_p" in decoding:
                    decoding["top_p"] = float(decoding["top_p"])
                if "top_k" in decoding:
                    decoding["top_k"] = int(decoding["top_k"])
            else:
                decoding.pop("temperature", None)
                decoding.pop("top_p", None)
                decoding.pop("top_k", None)

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
        system_message, user_message = base_messages
        decoding: Dict[str, Any] = dict(self.strategy.decoding or {})
        if preparation and preparation.get("decoding_override"):
            decoding.update(preparation["decoding_override"])  # type: ignore[arg-type]
        self._adjust_decoding(decoding)

        body_style = self._body_style()
        if body_style in {"json", "dsl", "kqml"}:
            return self._step_json(system_message, user_message, decoding)
        return self._step_with_trailer(system_message, user_message, decoding)

    # -- JSON mode -------------------------------------------------------
    def _step_json(
        self,
        system_message: Mapping[str, Any],
        user_message: Mapping[str, Any],
        decoding: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        max_attempts = 3
        errors: List[str] = []
        last_result: Optional[GenerationResult] = None
        raw_output = ""

        base_messages = [dict(system_message), dict(user_message)]

        for attempt in range(max_attempts):
            if attempt and errors:
                hint = _retry_instructions(errors)
                convo = _append_retry_hint(system_message, user_message, hint)
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

            try:
                candidate = json.loads(raw_output)
            except json.JSONDecodeError as exc:
                errors = [f"Invalid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})."]
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
            fallback.setdefault("content", {})["errors"] = list(errors)
        return repair_envelope(fallback), raw_output

    # -- trailer mode ----------------------------------------------------
    def _step_with_trailer(
        self,
        system_message: Mapping[str, Any],
        user_message: Mapping[str, Any],
        decoding: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        max_attempts = 3
        failure_codes: List[str] = []
        errors: List[str] = []
        last_output = ""
        totals: Dict[str, int] = {
            "tokens_used_total": 0,
            "tokens_reserved_total": 0,
            "body_tokens_total": 0,
            "trailer_tokens_total": 0,
            "tokens_body_overflow_total": 0,
            "tokens_trailer_overflow_total": 0,
            "body_budget": 0,
            "trailer_budget": 0,
        }
        pending_body: str = ""
        trailer_only_retry = False

        gen_kwargs = dict(decoding)
        max_new_tokens = int(gen_kwargs.pop("max_new_tokens", 512))
        do_sample = bool(gen_kwargs.pop("do_sample", False))
        temperature = gen_kwargs.pop("temperature", None)
        top_p = gen_kwargs.pop("top_p", None)
        top_k = gen_kwargs.pop("top_k", None)

        sampling_kwargs: Dict[str, Any] = {}
        if do_sample:
            if temperature is not None:
                sampling_kwargs["temperature"] = float(temperature)
            if top_p is not None:
                sampling_kwargs["top_p"] = float(top_p)
            if top_k is not None:
                sampling_kwargs["top_k"] = int(top_k)

        for attempt in range(max_attempts):
            if attempt and errors:
                hint = "\n".join(errors)
                convo = _append_retry_hint(system_message, user_message, hint)
            else:
                convo = [dict(system_message), dict(user_message)]

            result = generate_with_trailer(
                self.model,
                self.tokenizer,
                convo,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                **sampling_kwargs,
                **gen_kwargs,
            )
            last_output = result.text
            trailer_only_retry = False

            totals["tokens_used_total"] += max(int(result.tokens_used), 0)
            totals["tokens_reserved_total"] += max(int(result.tokens_reserved), 0)
            totals["body_tokens_total"] += max(int(result.body_tokens), 0)
            totals["trailer_tokens_total"] += max(int(result.trailer_tokens), 0)
            totals["tokens_body_overflow_total"] += max(int(result.tokens_body_overflow), 0)
            totals["tokens_trailer_overflow_total"] += max(
                int(result.tokens_trailer_overflow), 0
            )
            totals["body_budget"] = max(totals.get("body_budget", 0), int(result.body_budget))
            totals["trailer_budget"] = max(
                totals.get("trailer_budget", 0), int(result.trailer_budget)
            )

            extraction = extract_control_trailer(last_output)
            offsets = extraction.get("offsets") or {}
            json_start = offsets.get("json_start", -1)
            json_end = offsets.get("json_end", -1)
            trailer_json = (
                last_output[json_start:json_end]
                if json_start != -1 and json_end != -1 and json_end >= json_start
                else ""
            )

            if not extraction.get("ok"):
                error_code = extraction.get("error") or "UNKNOWN"
                failure_codes.append(error_code)
                errors = [_trailer_error_hint(error_code)]
                if error_code == "ERR_TRAILER_UNCLOSED":
                    pending_body = extraction.get("body") or pending_body
                continue

            payload = dict(extraction.get("payload") or {})
            validation = validate_control_payload(payload)
            if not validation.get("ok"):
                val_errors = list(validation.get("errors") or [])
                if not val_errors:
                    val_errors = ["Invalid control payload."]
                failure_codes.extend(val_errors)
                errors = val_errors
                continue

            payload = dict(validation.get("payload") or {})
            body_text = extraction.get("body") or ""
            if not body_text and pending_body:
                body_text = pending_body
                extraction = dict(extraction)
                extraction["body"] = body_text
                if extraction.get("offsets"):
                    offsets = dict(extraction["offsets"])
                else:
                    offsets = {"json_start": json_start, "json_end": json_end, "suffix_at_end": True}
                shift = len(body_text)
                offsets["json_start"] = (offsets.get("json_start", -1) + shift) if offsets.get("json_start", -1) != -1 else -1
                offsets["json_end"] = (offsets.get("json_end", -1) + shift) if offsets.get("json_end", -1) != -1 else -1
                extraction["offsets"] = offsets
                trailer_only_retry = True
                if body_text and not last_output.startswith(body_text):
                    last_output = f"{body_text}{last_output}"
            pending_body = body_text
            content = dict(payload.get("content") or {})
            if body_text.strip():
                content.setdefault("body", body_text.strip())
            payload["content"] = content

            telemetry = _telemetry_from(
                result,
                extraction,
                failure_codes,
                attempt,
                totals,
                trailer_only_retry,
            )
            payload["telemetry"] = telemetry

            envelope = envelope_from_payload(payload)
            content_block = dict(envelope.get("content") or {})
            control_meta = {
                "source": "control_trailer",
                "telemetry": telemetry,
                "raw_trailer": f"{CTRL_PREFIX}{trailer_json}{CTRL_SUFFIX}" if trailer_json else None,
                "body_preview": _truncate(body_text),
                "errors": list(failure_codes),
            }
            if control_meta["raw_trailer"] is None:
                control_meta.pop("raw_trailer")
            if not control_meta["errors"]:
                control_meta.pop("errors")
            content_block["control"] = control_meta
            envelope["content"] = content_block
            return repair_envelope(envelope), last_output

        fallback: Dict[str, Any] = {
            "tag": "[CONTACT]",
            "status": "NEED_PEER",
            "content": {
                "intent": "QUESTION",
                "message": "Unable to produce a valid control trailer.",
                "errors": list(failure_codes) or list(errors),
            },
        }
        return repair_envelope(fallback), last_output


__all__ = ["HFChatAgent", "CONTROL_TRAILER_GUIDE"]
