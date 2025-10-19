from __future__ import annotations

"""Utilities for loading Hugging Face causal language models."""

from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TINY_MODEL_ID = "roneneldan/TinyStories-1M"
MISTRAL_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"


def _resolve_dtype(dtype: Optional[str]):
    if dtype is None:
        return None
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return mapping.get(str(dtype).lower())


@lru_cache(maxsize=4)
def load_causal_lm(
    model_id: str,
    *,
    dtype: Optional[str] = None,
    trust_remote_code: bool = False,
    device_map: Optional[str] = "auto",
    low_cpu_mem_usage: bool = True,
) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = _resolve_dtype(dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    model.eval()
    return tokenizer, model


def build_inputs(tokenizer, messages: Sequence[Dict[str, str]], *, add_generation_prompt: bool = True):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            list(messages),
            return_tensors="pt",
            add_generation_prompt=add_generation_prompt,
        )
    parts: List[str] = []
    for message in messages:
        parts.append(f"{message['role'].upper()}: {message['content']}")
    prompt = "\n".join(parts)
    if add_generation_prompt:
        prompt += "\nASSISTANT:"
    return tokenizer(prompt, return_tensors="pt").input_ids


@torch.inference_mode()
def generate_chat_completion(
    tokenizer,
    model,
    messages: Sequence[Dict[str, str]],
    *,
    decoding: Optional[Dict[str, Any]] = None,
) -> str:
    decode_cfg = dict(decoding or {})
    max_new_tokens = int(decode_cfg.pop("max_new_tokens", 256))
    temperature = float(decode_cfg.pop("temperature", 0.0))
    do_sample = decode_cfg.pop("do_sample", None)
    if do_sample is None:
        do_sample = temperature > 0

    input_ids = build_inputs(tokenizer, messages)
    input_ids = input_ids.to(model.device)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
    }
    generation_kwargs.update(decode_cfg)

    output = model.generate(input_ids=input_ids, **generation_kwargs)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def generate_json_only(
    tokenizer,
    model,
    system_prompt: str | Sequence[Dict[str, str]],
    user_prompt: Optional[str] = None,
    *,
    decoding: Optional[Dict[str, Any]] = None,
) -> str:
    if isinstance(system_prompt, Sequence) and not isinstance(system_prompt, str):
        messages = list(system_prompt)
    else:
        messages = [
            {"role": "system", "content": str(system_prompt)},
            {"role": "user", "content": str(user_prompt or "")},
        ]
    return generate_chat_completion(tokenizer, model, messages, decoding=decoding)


__all__ = [
    "TINY_MODEL_ID",
    "MISTRAL_MODEL_ID",
    "load_causal_lm",
    "generate_chat_completion",
    "generate_json_only",
    "build_inputs",
]
