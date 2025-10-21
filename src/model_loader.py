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
TRAILER_RESERVE_FRACTION = 0.25


_DTYPE_ALIASES: Mapping[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


class SuffixStopper(StoppingCriteria):
    """Stop generation once the token sequence ends with the CTRL suffix."""

    def __init__(self, tokenizer: PreTrainedTokenizer, suffix: str) -> None:
        super().__init__()
        self._suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        if not self._suffix_ids:
            # Fallback for tokenizers that require a leading space
            self._suffix_ids = tokenizer.encode(" " + suffix, add_special_tokens=False)
        self.triggered = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **_: Any) -> bool:
        if not self._suffix_ids:
            return False
        sequence = input_ids[0].tolist()
        if len(sequence) < len(self._suffix_ids):
            return False
        if sequence[-len(self._suffix_ids) :] == self._suffix_ids:
            self.triggered = True
            return True
        return False

    def set_input_length(self, length: int) -> None:
        self._input_length = max(int(length), 0)


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


def _estimate_trailer_budget(tokenizer: PreTrainedTokenizer, max_new_tokens: int) -> int:
    if max_new_tokens <= 0:
        return 0
    try:
        template_tokens = tokenizer.encode(TRAILER_TEMPLATE, add_special_tokens=False)
    except TypeError:
        template_tokens = tokenizer.encode(TRAILER_TEMPLATE)
    except Exception:
        template_tokens = None
    if hasattr(template_tokens, "__len__"):
        template_len = len(template_tokens)  # type: ignore[arg-type]
    else:
        template_len = 0
    if template_len <= 0:
        template_len = max(len(TRAILER_TEMPLATE) // 4, 1)

    fractional_reserve = max(int(max_new_tokens * TRAILER_RESERVE_FRACTION), 0)
    trailer_budget = max(template_len, fractional_reserve)
    return min(max_new_tokens, trailer_budget)


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
    do_sample: bool = False,
    **generate_kwargs: Any,
) -> GenerationResult:
    input_ids = build_inputs(tokenizer, prompt, add_generation_prompt=True).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    stopper = SuffixStopper(tokenizer, CTRL_SUFFIX)
    stopping = StoppingCriteriaList([stopper])
    stopper.set_input_length(int(input_ids.shape[-1]))

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_token_id = tokenizer.eos_token_id

    sampling_kwargs: Dict[str, Any] = {}
    temperature = generate_kwargs.pop("temperature", None)
    top_p = generate_kwargs.pop("top_p", None)
    top_k = generate_kwargs.pop("top_k", None)
    if do_sample:
        if temperature is not None:
            sampling_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            sampling_kwargs["top_p"] = float(top_p)
        if top_k is not None:
            sampling_kwargs["top_k"] = int(top_k)

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
        stopping_criteria=stopping,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        **sampling_kwargs,
        **generate_kwargs,
    )

    gen_tokens = generated[0][input_ids.shape[-1] :]
    eos_hit = bool(len(gen_tokens) and eos_token_id is not None and int(gen_tokens[-1]) == int(eos_token_id))
    decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)

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

    trailer_budget = _estimate_trailer_budget(tokenizer, int(max_new_tokens))
    body_budget = max(int(max_new_tokens) - trailer_budget, 0)

    stop_reason = "ctrl_suffix" if stopper.triggered else ("eos" if eos_hit else "max_new_tokens")

    tokens_used = int(gen_tokens.shape[-1])
    overflow_tokens = max(0, tokens_used - int(max_new_tokens))

    return GenerationResult(
        text=decoded,
        stop_reason=stop_reason,
        tokens_used=tokens_used,
        overflow_tokens=overflow_tokens,
        has_tail=has_tail,
        trailer_offset=trailer_offset,
        input_tokens=int(input_ids.shape[-1]),
        max_new_tokens=int(max_new_tokens),
        tokens_reserved=int(max_new_tokens),
        body_tokens=body_tokens,
        trailer_tokens=trailer_tokens,
        tokens_body_overflow=max(body_tokens - body_budget, 0),
        tokens_trailer_overflow=max(trailer_tokens - trailer_budget, 0),
        suffix_triggered=stopper.triggered,
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
    "SuffixStopper",
    "build_inputs",
    "generate_json_only",
    "generate_with_trailer",
    "load_causal_lm",
    "load_model_and_tokenizer",
    "_render_chat",
]
