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


TRAILER_TEMPLATE = '<<<CTRL{"tag":"","status":"","content":{},"final_solution":{}}CTRL>>>'
DEFAULT_TRAILER_MARGIN = 64


_DTYPE_ALIASES: Mapping[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _token_length(tokenizer: PreTrainedTokenizer, text: str) -> int:
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        tokens = tokenizer.encode(text)  # type: ignore[arg-type]
    except Exception:
        tokens = None

    if isinstance(tokens, torch.Tensor):
        return int(tokens.shape[-1])
    if hasattr(tokens, "input_ids"):
        candidate = getattr(tokens, "input_ids")
        if hasattr(candidate, "shape"):
            return int(candidate.shape[-1])
        if hasattr(candidate, "__len__"):
            return len(candidate)  # type: ignore[arg-type]
    if isinstance(tokens, (list, tuple)):
        return len(tokens)
    if hasattr(tokens, "__len__"):
        try:
            return len(tokens)  # type: ignore[arg-type]
        except TypeError:
            return 0
    return 0


def estimate_trailer_token_budget(
    tokenizer: PreTrainedTokenizer, *, margin: int = DEFAULT_TRAILER_MARGIN
) -> int:
    """Estimate trailer token budget for the CTRL envelope with configurable margin."""

    base = getattr(tokenizer, "_ctrl_trailer_base_tokens", None)
    if not isinstance(base, int):
        base = _token_length(tokenizer, TRAILER_TEMPLATE)
        if base == 0:
            base = _token_length(tokenizer, " " + TRAILER_TEMPLATE)
        setattr(tokenizer, "_ctrl_trailer_base_tokens", base)

    margin = max(int(margin), 0)
    return base + margin


class SuffixStopper(StoppingCriteria):
    """Stop generation once the token sequence ends with the CTRL suffix."""

    def __init__(self, tokenizer: PreTrainedTokenizer, suffix: str) -> None:
        super().__init__()
        self._suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        if not self._suffix_ids:
            # Fallback for tokenizers that require a leading space
            self._suffix_ids = tokenizer.encode(" " + suffix, add_special_tokens=False)
        self.triggered = False
        self._input_length = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **_: Any) -> bool:
        if not self._suffix_ids:
            return False
        sequence = input_ids[0].tolist()
        if self._input_length:
            sequence = sequence[self._input_length :]
        if len(sequence) < len(self._suffix_ids):
            return False
        if sequence[-len(self._suffix_ids) :] == self._suffix_ids:
            self.triggered = True
            return True
        return False

    def set_input_length(self, length: int) -> None:
        self._input_length = max(int(length), 0)


@dataclass
class GenerationResult:
    text: str
    stop_reason: str
    tokens_used: int
    overflow_tokens: int
    has_tail: bool
    trailer_offset: int
    input_tokens: int = 0
    max_new_tokens: int = 0
    body_budget: int = 0
    trailer_budget: int = 0
    tokens_reserved: int = 0
    body_tokens: int = 0
    trailer_tokens: int = 0
    tokens_body_overflow: int = 0
    tokens_trailer_overflow: int = 0
    suffix_triggered: bool = False


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
    used = int(gen_tokens.shape[-1])
    overflow = max(0, used - int(max_new_tokens))
    return GenerationResult(
        text=text,
        stop_reason=stop_reason,
        tokens_used=used,
        overflow_tokens=overflow,
        has_tail=has_tail,
        trailer_offset=trailer_offset,
        input_tokens=int(input_ids.shape[-1]),
        max_new_tokens=int(max_new_tokens),
        tokens_reserved=int(max_new_tokens),
    )


@torch.inference_mode()
def generate_with_trailer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: Sequence[Dict[str, str]] | str,
    *,
    body_token_budget: Optional[int] = None,
    trailer_token_budget: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    do_sample: bool = False,
    trailer_margin: int = DEFAULT_TRAILER_MARGIN,
    **generate_kwargs: Any,
) -> GenerationResult:
    input_ids = build_inputs(tokenizer, prompt, add_generation_prompt=True).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    stopper = SuffixStopper(tokenizer, CTRL_SUFFIX)
    stopper.set_input_length(int(input_ids.shape[-1]))
    stopping = StoppingCriteriaList([stopper])

    if trailer_token_budget is None:
        trailer_token_budget = estimate_trailer_token_budget(tokenizer, margin=trailer_margin)
    trailer_token_budget = max(int(trailer_token_budget), 0)

    if body_token_budget is None:
        if max_new_tokens is not None:
            total = max(int(max_new_tokens), 0)
            body_token_budget = max(total - trailer_token_budget, 0)
        else:
            # Preserve historical behaviour (default total of 512 tokens).
            total_default = 512
            body_token_budget = max(total_default - trailer_token_budget, 0)
    body_token_budget = max(int(body_token_budget), 0)

    total_budget = int(body_token_budget + trailer_token_budget)
    generate_kwargs.pop("max_new_tokens", None)

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_token_id = tokenizer.eos_token_id

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=total_budget,
        do_sample=bool(do_sample),
        stopping_criteria=stopping,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        **generate_kwargs,
    )

    gen_tokens = generated[0][input_ids.shape[-1] :]
    tokens_used = int(gen_tokens.shape[-1])
    eos_hit = bool(tokens_used and eos_token_id is not None and int(gen_tokens[-1]) == int(eos_token_id))

    decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    trimmed = decoded.rstrip()
    suffix_at_end = trimmed.endswith(CTRL_SUFFIX) and trimmed == decoded
    has_tail = not suffix_at_end

    prefix_idx = decoded.rfind(CTRL_PREFIX)
    trailer_text = decoded[prefix_idx:] if prefix_idx != -1 else ""
    body_text = decoded[:prefix_idx] if prefix_idx != -1 else decoded

    trailer_offset = -1
    if suffix_at_end and prefix_idx != -1:
        trailer_offset = prefix_idx + len(CTRL_PREFIX)

    body_tokens = _token_length(tokenizer, body_text) if body_text else 0
    trailer_tokens = _token_length(tokenizer, trailer_text) if trailer_text else 0

    stop_reason: str
    if stopper.triggered:
        stop_reason = "suffix"
    elif tokens_used >= total_budget:
        stop_reason = "max_new_tokens"
    elif eos_hit:
        stop_reason = "eos_token"
    else:
        stop_reason = "unknown"

    overflow_tokens = max(0, tokens_used - total_budget)
    tokens_body_overflow = max(0, body_tokens - body_token_budget)
    tokens_trailer_overflow = max(0, trailer_tokens - trailer_token_budget)

    return GenerationResult(
        text=decoded,
        stop_reason=stop_reason,
        tokens_used=tokens_used,
        overflow_tokens=overflow_tokens,
        has_tail=has_tail,
        trailer_offset=trailer_offset,
        input_tokens=int(input_ids.shape[-1]),
        max_new_tokens=total_budget,
        body_budget=body_token_budget,
        trailer_budget=trailer_token_budget,
        tokens_reserved=total_budget,
        body_tokens=body_tokens,
        trailer_tokens=trailer_tokens,
        tokens_body_overflow=tokens_body_overflow,
        tokens_trailer_overflow=tokens_trailer_overflow,
        suffix_triggered=stopper.triggered,
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
    "estimate_trailer_token_budget",
    "SuffixStopper",
    "build_inputs",
    "generate_json_only",
    "generate_with_trailer",
    "load_causal_lm",
    "load_model_and_tokenizer",
    "_render_chat",
]
