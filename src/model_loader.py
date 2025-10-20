from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


TINY_REPO = "roneneldan/TinyStories-1M"
TINY_TOKENIZER = "EleutherAI/gpt-neo-125M"


_DTYPE_ALIASES: Mapping[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _resolve_dtype(dtype: Optional[str]) -> Optional[torch.dtype]:
    if not dtype:
        return None
    return _DTYPE_ALIASES.get(dtype.lower())


def load_model_and_tokenizer(
    repo_id: str,
    *,
    tokenizer_id: Optional[str] = None,
    dtype: Optional[str] = None,
    quant: Optional[str] = None,
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Load a causal LM and tokenizer for chat-style generation."""

    tokenizer_ref = tokenizer_id or repo_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref)

    model_kwargs: Dict[str, Any] = {"device_map": "auto"}
    target_dtype = _resolve_dtype(dtype)
    if target_dtype is not None:
        model_kwargs["torch_dtype"] = target_dtype

    if quant == "4bit":
        model_kwargs.update(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quant == "8bit":
        model_kwargs.update(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(repo_id, **model_kwargs)
    model.eval()
    return tokenizer, model


def _has_chat_template(tokenizer: PreTrainedTokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


def build_inputs(
    tokenizer: PreTrainedTokenizer,
    messages_or_text: Sequence[Dict[str, str]] | str,
    *,
    add_generation_prompt: bool = True,
) -> torch.Tensor:
    """Format messages according to the tokenizer's template (if any)."""

    if _has_chat_template(tokenizer):
        if not isinstance(messages_or_text, Sequence) or not messages_or_text:
            raise TypeError("Chat templates require a list of messages.")
        return tokenizer.apply_chat_template(
            list(messages_or_text),
            return_tensors="pt",
            add_generation_prompt=add_generation_prompt,
        )

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


@torch.inference_mode()
def _generate(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    messages: Sequence[Dict[str, str]] | str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    do_sample: Optional[bool] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    **extra: Any,
) -> str:
    if do_sample is None:
        do_sample = bool(temperature and float(temperature) > 0.0)

    input_ids = build_inputs(tokenizer, messages, add_generation_prompt=True).to(model.device)
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens or 256),
        "temperature": float(temperature or 0.0),
        "do_sample": bool(do_sample),
    }
    if top_p is not None:
        gen_kwargs["top_p"] = float(top_p)
    if top_k is not None:
        gen_kwargs["top_k"] = int(top_k)
    gen_kwargs.update(extra)

    output = model.generate(input_ids=input_ids, **gen_kwargs)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def generate_json_only(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    prompt_or_messages: Sequence[Dict[str, str]] | str,
    *,
    user_prompt: Optional[str] = None,
    decoding: Optional[Dict[str, Any]] = None,
    **legacy_kwargs: Any,
) -> str:
    """Generate text biased toward JSON envelopes."""

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

    max_new_tokens = decode_cfg.pop("max_new_tokens", 256)
    temperature = decode_cfg.pop("temperature", 0.0)
    do_sample = decode_cfg.pop("do_sample", None)

    return _generate(
        tokenizer,
        model,
        messages,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        do_sample=do_sample,
        **decode_cfg,
    )


__all__ = [
    "TINY_REPO",
    "TINY_TOKENIZER",
    "build_inputs",
    "generate_json_only",
    "load_model_and_tokenizer",
]

