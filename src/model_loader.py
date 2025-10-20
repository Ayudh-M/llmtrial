from __future__ import annotations

"""Utilities for loading Hugging Face causal language models.

This module primarily targets real Hugging Face models.  However, our CI
environment does not always have network access which makes downloading the
reference tiny model flaky.  To ensure the tiny-model tests still exercise the
pipeline we provide a lightweight deterministic fallback that mimics the
transformers API.  The fallback is only activated when the tiny model cannot be
downloaded and should not affect real deployments where the genuine model is
available.
"""

from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import warnings

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
    try:
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
    except Exception as exc:  # pragma: no cover - we test the fallback path instead
        if model_id != TINY_MODEL_ID:
            raise
        warnings.warn(
            "Falling back to offline tiny model stub because loading"
            f" '{TINY_MODEL_ID}' failed with: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _load_offline_tiny_model()


OFFLINE_TINY_COMPLETION = "Hello from the offline tiny model."


class _OfflineTinyTokenizer:
    """Minimal tokenizer used when the real tiny model cannot be downloaded."""

    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.special_tokens = {self.pad_token, self.eos_token}
        self._token_to_id = {self.pad_token: 0, self.eos_token: 1}
        self._id_to_token = {0: self.pad_token, 1: self.eos_token}

    def _tokenize(self, text: str) -> List[str]:
        text = text.replace("\n", " \n ")
        tokens = text.split()
        if not tokens:
            tokens = [self.pad_token]
        return tokens

    def _encode_tokens(self, tokens: Sequence[str]) -> List[int]:
        ids: List[int] = []
        for token in tokens:
            if token not in self._token_to_id:
                idx = len(self._token_to_id)
                self._token_to_id[token] = idx
                self._id_to_token[idx] = token
            ids.append(self._token_to_id[token])
        return ids

    def __call__(self, text: str, return_tensors: Optional[str] = None):
        tokens = self._tokenize(text)
        ids = self._encode_tokens(tokens)
        tensor = torch.tensor([ids], dtype=torch.long)
        if return_tensors == "pt":
            return SimpleNamespace(input_ids=tensor)
        return tensor

    def apply_chat_template(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        return_tensors: Optional[str] = None,
        add_generation_prompt: bool = True,
    ):
        lines: List[str] = [f"{m['role'].upper()}: {m['content']}" for m in messages]
        if add_generation_prompt:
            lines.append("ASSISTANT:")
        prompt = "\n".join(lines)
        encoded = self(prompt, return_tensors=return_tensors)
        if isinstance(encoded, SimpleNamespace):
            return encoded.input_ids
        return encoded

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        tokens: List[str] = []
        for idx in ids:
            token = self._id_to_token.get(int(idx), "")
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        return " ".join(tokens).replace(" \n ", "\n").strip()


class _OfflineTinyModel:
    """Tiny deterministic model used for offline testing."""

    def __init__(self, tokenizer: _OfflineTinyTokenizer) -> None:
        self.tokenizer = tokenizer
        self.device = torch.device("cpu")
        self.offline_stub = True

    def to(self, device: torch.device | str):
        self.device = torch.device(device)
        return self

    def eval(self):  # pragma: no cover - simple stub
        return self

    def generate(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        response_ids = self.tokenizer(OFFLINE_TINY_COMPLETION + f" {self.tokenizer.eos_token}")
        response_tensor = torch.as_tensor(response_ids, dtype=torch.long)
        response_tensor = response_tensor.to(self.device)
        input_ids = input_ids.to(self.device)
        if response_tensor.dim() == 1:
            response_tensor = response_tensor.unsqueeze(0)
        return torch.cat([input_ids, response_tensor[:, : response_tensor.shape[1]]], dim=1)


def _load_offline_tiny_model() -> Tuple[_OfflineTinyTokenizer, _OfflineTinyModel]:
    tokenizer = _OfflineTinyTokenizer()
    model = _OfflineTinyModel(tokenizer)
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
