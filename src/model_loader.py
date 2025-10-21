from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from jinja2 import TemplateError

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from .control_trailer import CTRL_PREFIX, CTRL_SUFFIX


TINY_REPO = "roneneldan/TinyStories-1M"
TINY_TOKENIZER = "EleutherAI/gpt-neo-125M"


TRAILER_RESERVE_FRACTION = 0.25


_DTYPE_ALIASES: Mapping[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


class SuffixStop(StoppingCriteria):
    """Stop generation once the decoded text contains the CTRL suffix."""

    def __init__(self, tokenizer: PreTrainedTokenizer, suffix: str, lookback_chars: int = 160) -> None:
        super().__init__()
        self._tokenizer = tokenizer
        self._suffix = suffix
        self._lookback = max(int(lookback_chars), len(suffix))
        self.triggered = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **_: Any) -> bool:
        if input_ids.numel() == 0:
            return False
        ids = input_ids[0].tolist()
        tail = self._tokenizer.decode(ids[-512:], skip_special_tokens=False)
        if self._suffix in tail[-self._lookback :]:
            self.triggered = True
            return True
        return False


def _safe_token_length(tokenizer: PreTrainedTokenizer, text: str) -> int:
    if not text:
        return 0
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        tokens = tokenizer.encode(text)
    except Exception:
        tokens = None
    if tokens is None:
        stripped = text.strip()
        return len(stripped.split()) if stripped else 0
    if hasattr(tokens, "input_ids") and hasattr(tokens.input_ids, "__len__"):
        return len(tokens.input_ids)  # type: ignore[arg-type]
    if hasattr(tokens, "__len__"):
        return len(tokens)  # type: ignore[arg-type]
    return 0


@dataclass
class GenerationResult:
    text: str
    stop_reason: str
    tokens_used: int
    overflow_tokens: int = 0
    has_tail: bool = False
    trailer_offset: int = -1
    input_tokens: int = 0
    max_new_tokens: int = 0
    tokens_reserved: int = 0
    body_tokens: int = 0
    trailer_tokens: int = 0
    tokens_body_overflow: int = 0
    tokens_trailer_overflow: int = 0
    suffix_triggered: bool = False
    body_budget: int = 0
    trailer_budget: int = 0


def _resolve_dtype(dtype: Optional[str]) -> Optional[torch.dtype]:
    if not dtype:
        return None
    return _DTYPE_ALIASES.get(dtype.lower())


def load_model_and_tokenizer(
    model_name: str,
    *,
    tokenizer_name: Optional[str] = None,
    dtype: Optional[str] = "bf16",
    **extra: Any,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a causal LM and matching tokenizer."""

    tok_ref = tokenizer_name or model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_ref, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    kwargs: Dict[str, Any] = {"device_map": "auto"}
    resolved = _resolve_dtype(dtype)
    if resolved is not None:
        kwargs["torch_dtype"] = resolved
    kwargs.update(extra)

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **kwargs)
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config") and tokenizer.pad_token_id is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer


def load_causal_lm(
    model_name: str,
    *,
    tokenizer_name: Optional[str] = None,
    dtype: Optional[str] = "bf16",
    **extra: Any,
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Backward-compatible helper returning (tokenizer, model)."""

    model, tokenizer = load_model_and_tokenizer(
        model_name, tokenizer_name=tokenizer_name, dtype=dtype, **extra
    )
    return tokenizer, model


def _has_chat_template(tokenizer: PreTrainedTokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


def _merge_system_messages(messages: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    merged: List[Dict[str, str]] = []
    system_chunks: List[str] = []

    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if role.lower() == "system":
            if content:
                system_chunks.append(content)
            continue

        if system_chunks:
            prefix = "\n\n".join(chunk for chunk in system_chunks if chunk)
            if prefix:
                content = f"{prefix}\n\n{content}" if content else prefix
            system_chunks.clear()

        merged.append({"role": role or "user", "content": content})

    if system_chunks:
        prefix = "\n\n".join(chunk for chunk in system_chunks if chunk)
        if merged:
            first = dict(merged[0])
            existing = first.get("content", "")
            first["content"] = f"{prefix}\n\n{existing}" if existing else prefix
            merged[0] = first
        elif prefix:
            merged.append({"role": "user", "content": prefix})

    return merged


def _render_chat(tokenizer: PreTrainedTokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return f"SYSTEM: {system_prompt}\nUSER: {user_prompt}\nASSISTANT:"


def build_inputs(
    tokenizer: PreTrainedTokenizer,
    messages_or_text: Sequence[Dict[str, str]] | str,
    *,
    add_generation_prompt: bool = True,
) -> torch.Tensor:
    if _has_chat_template(tokenizer):
        if not isinstance(messages_or_text, Sequence) or not messages_or_text:
            raise TypeError("Chat templates require a list of messages.")

        prepared = [
            {
                "role": str(message.get("role", "")),
                "content": str(message.get("content", "")),
            }
            for message in messages_or_text  # type: ignore[arg-type]
        ]

        candidates = [prepared]
        merged = _merge_system_messages(prepared)
        if merged and merged != prepared:
            candidates.append(merged)

        last_attempt = prepared
        for convo in candidates:
            try:
                return tokenizer.apply_chat_template(
                    convo,
                    return_tensors="pt",
                    add_generation_prompt=add_generation_prompt,
                )
            except (TemplateError, ValueError, TypeError):
                last_attempt = convo
                continue

        messages_or_text = last_attempt  # fall back to plain formatting

    if isinstance(messages_or_text, Sequence) and messages_or_text and isinstance(messages_or_text[0], dict):
        lines = []
        for message in messages_or_text:  # type: ignore[assignment]
            role = message.get("role", "user")
            content = message.get("content", "")
            lines.append(f"{role.upper()}: {content}")
        prompt = "\n".join(lines)
        if add_generation_prompt:
            prompt += "\nASSISTANT:"
    else:
        prompt = str(messages_or_text)

    encoded = tokenizer(prompt, return_tensors="pt")
    return encoded.input_ids


def _decode_generated(
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor,
    generated: torch.Tensor,
    *,
    stop_reason: str,
    max_new_tokens: int,
    has_tail: bool = False,
    trailer_offset: int = -1,
) -> GenerationResult:
    gen_tokens = generated[0][input_ids.shape[-1] :]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return GenerationResult(
        text=text,
        stop_reason=stop_reason,
        tokens_used=int(gen_tokens.shape[-1]),
        overflow_tokens=max(0, int(gen_tokens.shape[-1]) - int(max_new_tokens)),
        has_tail=has_tail,
        trailer_offset=trailer_offset,
        input_tokens=int(input_ids.shape[-1]),
        max_new_tokens=int(max_new_tokens),
    )


@torch.inference_mode()
def generate_with_trailer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: Sequence[Dict[str, str]] | str,
    *,
    max_new_tokens: int = 512,
    **generate_kwargs: Any,
) -> GenerationResult:
    params: Dict[str, Any] = dict(generate_kwargs)

    requested_max = int(params.pop("max_new_tokens", max_new_tokens))
    trailer_budget = int(
        params.pop(
            "trailer_budget",
            max(int(requested_max * TRAILER_RESERVE_FRACTION), 96),
        )
    )
    body_budget = int(params.pop("body_budget", max(requested_max - trailer_budget, 0)))
    ensured_max = max(requested_max, body_budget + trailer_budget)
    max_new_tokens = ensured_max

    min_new_tokens = int(params.pop("min_new_tokens", max(min(64, max_new_tokens // 2), 0)))
    raw_salvage = params.pop("salvage_max_new_tokens", None)
    salvage_max_tokens = max(
        int(raw_salvage) if raw_salvage is not None else max(trailer_budget, 64),
        1,
    )

    do_sample = bool(params.pop("do_sample", True))

    raw_temperature = params.pop("temperature", None)
    temperature = float(raw_temperature) if raw_temperature is not None else 0.7

    raw_top_p = params.pop("top_p", None)
    top_p = float(raw_top_p) if raw_top_p is not None else 0.9

    raw_top_k = params.pop("top_k", None)
    top_k = int(raw_top_k) if raw_top_k is not None else None

    raw_rep_penalty = params.pop("repetition_penalty", None)
    repetition_penalty = float(raw_rep_penalty) if raw_rep_penalty is not None else 1.05

    input_ids = build_inputs(tokenizer, prompt, add_generation_prompt=True).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    stopper = SuffixStop(tokenizer, CTRL_SUFFIX)
    stopping = StoppingCriteriaList([stopper])

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_token_id = tokenizer.eos_token_id
    bad_words_ids: Optional[List[List[int]]] = None
    if eos_token_id is not None:
        bad_words_ids = [[int(eos_token_id)]]

    generation_args: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
        "stopping_criteria": stopping,
        "pad_token_id": pad_token_id,
    }

    if do_sample:
        generation_args["temperature"] = temperature
        generation_args["top_p"] = top_p
        if top_k is not None:
            generation_args["top_k"] = int(top_k)
    elif top_k is not None:
        generation_args["top_k"] = int(top_k)

    if bad_words_ids is not None:
        generation_args["bad_words_ids"] = bad_words_ids

    generation_args.update(params)

    generated = model.generate(**generation_args)

    gen_tokens = generated[0][input_ids.shape[-1] :]
    decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    decoded_tail = tokenizer.decode(generated[0][-512:], skip_special_tokens=False)

    suffix_triggered = stopper.triggered or CTRL_SUFFIX in decoded_tail

    salvage_tokens_used = 0
    salvage_invocations = 0
    salvage_last_token: Optional[int] = None

    def _run_salvage(prompt_text: str) -> Tuple[str, bool]:
        nonlocal salvage_tokens_used, salvage_last_token, salvage_invocations
        salvage_input = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
        salvage_mask = torch.ones_like(salvage_input)
        salvage_stop = SuffixStop(tokenizer, CTRL_SUFFIX)
        salvage_args: Dict[str, Any] = {
            "input_ids": salvage_input,
            "attention_mask": salvage_mask,
            "max_new_tokens": salvage_max_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "stopping_criteria": StoppingCriteriaList([salvage_stop]),
            "pad_token_id": pad_token_id,
        }
        if top_k is not None:
            salvage_args["top_k"] = int(top_k)
        if bad_words_ids is not None:
            salvage_args["bad_words_ids"] = bad_words_ids

        salvage_invocations += 1
        salvage_out = model.generate(**salvage_args)
        salvage_gen = salvage_out[0][salvage_input.shape[-1] :]
        salvage_tokens_used += int(salvage_gen.shape[-1])
        if salvage_gen.numel():
            salvage_last_token = int(salvage_gen[-1])
        salvage_text = tokenizer.decode(salvage_gen, skip_special_tokens=True)
        triggered = salvage_stop.triggered or CTRL_SUFFIX in salvage_text
        return salvage_text, triggered

    if CTRL_PREFIX not in decoded and decoded.strip():
        trailer_prompt = (
            "You wrote the following response body:\n"
            f"{decoded.rstrip()}\n\n"
            "Now output ONLY the control trailer in the exact <<<CTRL{...}CTRL>>> format. "
            "Begin immediately with <<<CTRL{ and do not repeat the body."
        )
        trailer_text, trailer_triggered = _run_salvage(trailer_prompt)
        if trailer_text:
            prefix_pos = trailer_text.find(CTRL_PREFIX)
            if prefix_pos != -1:
                trailer_text = trailer_text[prefix_pos:]
            trailer_text = trailer_text.lstrip()
            if CTRL_SUFFIX in trailer_text:
                end_pos = trailer_text.find(CTRL_SUFFIX) + len(CTRL_SUFFIX)
                trailer_text = trailer_text[:end_pos]
            if trailer_text:
                if decoded and not decoded.endswith("\n"):
                    decoded = f"{decoded}\n"
                decoded = f"{decoded}{trailer_text}"
                suffix_triggered = suffix_triggered or trailer_triggered or CTRL_SUFFIX in trailer_text

    if CTRL_PREFIX in decoded and CTRL_SUFFIX not in decoded:
        trailer_start = decoded.rfind(CTRL_PREFIX)
        partial_trailer = decoded[trailer_start:]
        salvage_prompt = (
            "Continue exactly from the partial control trailer below. "
            "Output only the remaining JSON and the literal CTRL>>>.\n"
            f"{partial_trailer}"
        )
        salvage_text, salvage_triggered = _run_salvage(salvage_prompt)
        decoded = f"{decoded}{salvage_text}"
        suffix_triggered = suffix_triggered or salvage_triggered or CTRL_SUFFIX in (
            partial_trailer + salvage_text
        )

    stripped = decoded.rstrip()
    suffix_at_end = stripped.endswith(CTRL_SUFFIX)
    has_tail = not suffix_at_end or len(decoded) != len(stripped)
    trailer_prefix_pos = decoded.rfind(CTRL_PREFIX)
    trailer_offset = -1
    if trailer_prefix_pos != -1:
        brace_pos = decoded.find("{", trailer_prefix_pos)
        if brace_pos != -1:
            trailer_offset = brace_pos

    trailer_text = decoded[trailer_prefix_pos:] if trailer_prefix_pos != -1 else ""
    body_text = decoded[:trailer_prefix_pos] if trailer_prefix_pos != -1 else decoded

    body_tokens = _safe_token_length(tokenizer, body_text)
    trailer_tokens = _safe_token_length(tokenizer, trailer_text)

    tokens_used = int(gen_tokens.shape[-1]) + salvage_tokens_used
    tokens_reserved = int(max_new_tokens) + salvage_invocations * salvage_max_tokens
    overflow_tokens = max(0, tokens_used - tokens_reserved)

    eos_stopped = False
    if eos_token_id is not None:
        eos_stopped = bool(gen_tokens.numel() and int(gen_tokens[-1]) == int(eos_token_id))
        if salvage_last_token is not None:
            eos_stopped = eos_stopped or salvage_last_token == int(eos_token_id)

    stop_reason = "suffix" if suffix_triggered and suffix_at_end else (
        "eos" if eos_stopped else ("max_new_tokens" if tokens_used >= max_new_tokens else "other")
    )

    return GenerationResult(
        text=decoded,
        stop_reason=stop_reason,
        tokens_used=tokens_used,
        overflow_tokens=overflow_tokens,
        has_tail=has_tail,
        trailer_offset=trailer_offset,
        input_tokens=int(input_ids.shape[-1]),
        max_new_tokens=int(max_new_tokens),
        tokens_reserved=tokens_reserved,
        body_tokens=body_tokens,
        trailer_tokens=trailer_tokens,
        tokens_body_overflow=max(body_tokens - body_budget, 0),
        tokens_trailer_overflow=max(trailer_tokens - trailer_budget, 0),
        suffix_triggered=suffix_triggered,
        body_budget=body_budget,
        trailer_budget=trailer_budget,
    )


@torch.inference_mode()
def generate_json_only(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    prompt_or_messages: Sequence[Dict[str, str]] | str,
    *,
    user_prompt: Optional[str] = None,
    decoding: Optional[Dict[str, Any]] = None,
    **legacy_kwargs: Any,
) -> GenerationResult:
    if isinstance(prompt_or_messages, Sequence) and prompt_or_messages and isinstance(prompt_or_messages[0], dict):
        messages = list(prompt_or_messages)  # type: ignore[arg-type]
    else:
        system_message = str(prompt_or_messages)
        user_message = str(user_prompt or "")
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    decode_cfg: Dict[str, Any] = dict(decoding or {})
    decode_cfg.update(legacy_kwargs)

    max_new_tokens = int(decode_cfg.pop("max_new_tokens", 256))
    do_sample = bool(decode_cfg.pop("do_sample", False))
    temperature = float(decode_cfg.pop("temperature", 0.0)) if do_sample else 0.0
    top_p = decode_cfg.pop("top_p", None)
    top_k = decode_cfg.pop("top_k", None)

    input_ids = build_inputs(tokenizer, messages, add_generation_prompt=True).to(model.device)
    attention_mask = torch.ones_like(input_ids)

    generate_args: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample and temperature:
        generate_args["temperature"] = float(temperature)
    if top_p is not None:
        generate_args["top_p"] = float(top_p)
    if top_k is not None:
        generate_args["top_k"] = int(top_k)
    generate_args.update(decode_cfg)

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_token_id = tokenizer.eos_token_id
    if pad_token_id is not None:
        generate_args.setdefault("pad_token_id", pad_token_id)
    if eos_token_id is not None:
        generate_args.setdefault("eos_token_id", eos_token_id)

    generated = model.generate(**generate_args)
    gen_tokens = generated[0][input_ids.shape[-1] :]
    eos_hit = bool(len(gen_tokens) and eos_token_id is not None and int(gen_tokens[-1]) == int(eos_token_id))
    stop_reason = "eos" if eos_hit else "length"

    return GenerationResult(
        text=tokenizer.decode(gen_tokens, skip_special_tokens=True),
        stop_reason=stop_reason,
        tokens_used=int(gen_tokens.shape[-1]),
        overflow_tokens=max(0, int(gen_tokens.shape[-1]) - max_new_tokens),
        has_tail=False,
        trailer_offset=-1,
        input_tokens=int(input_ids.shape[-1]),
        max_new_tokens=max_new_tokens,
    )


__all__ = [
    "TINY_REPO",
    "TINY_TOKENIZER",
    "GenerationResult",
    "SuffixStop",
    "build_inputs",
    "generate_json_only",
    "generate_with_trailer",
    "load_causal_lm",
    "load_model_and_tokenizer",
    "_render_chat",
]
