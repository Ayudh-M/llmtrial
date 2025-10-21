from __future__ import annotations

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

from .control_trailer import CTRL_SUFFIX


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


class ControlTrailerStoppingCriteria(StoppingCriteria):
    """Stop generation once the token sequence ends with the CTRL suffix."""

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()
        self.triggered = False
        self._suffix_tokens = tokenizer.encode(CTRL_SUFFIX, add_special_tokens=False)
        if not self._suffix_tokens:
            # Fallback for tokenizers that require a leading space.
            self._suffix_tokens = tokenizer.encode(" " + CTRL_SUFFIX, add_special_tokens=False)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        if not self._suffix_tokens:
            return False
        sequence = input_ids[0].tolist()
        suffix = self._suffix_tokens
        if len(sequence) < len(suffix):
            return False
        if sequence[-len(suffix) :] == suffix:
            self.triggered = True
            return True
        return False


@dataclass
class GenerationResult:
    """Container for decoded text plus generation telemetry."""

    text: str
    stop_reason: str
    new_tokens: int
    input_tokens: int
    max_new_tokens: int
    trailer_triggered: bool
    body_budget: int
    reserved_tokens: int


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
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

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
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config") and tokenizer.pad_token_id is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return tokenizer, model


def _has_chat_template(tokenizer: PreTrainedTokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


def _merge_system_messages(messages: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    """Merge leading system messages into the first non-system message.

    Some chat templates (e.g. Mistral) reject consecutive system messages or
    conversations that do not strictly alternate `user`/`assistant` roles. When
    that happens we fold system instructions into the first user turn so that
    the rendered conversation satisfies the template requirements.
    """

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
    stop_on_trailer: bool = True,
    trailer_reserved_tokens: int = 80,
    **extra: Any,
) -> GenerationResult:
    if do_sample is None:
        do_sample = bool(temperature and float(temperature) > 0.0)

    input_ids = build_inputs(tokenizer, messages, add_generation_prompt=True).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens or 256),
        "do_sample": bool(do_sample),
    }
    if do_sample:
        gen_kwargs["temperature"] = float(temperature or 0.0)
    if top_p is not None:
        gen_kwargs["top_p"] = float(top_p)
    if top_k is not None:
        gen_kwargs["top_k"] = int(top_k)
    gen_kwargs.update(extra)

    if tokenizer.pad_token_id is not None:
        gen_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def generate_json_only(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    prompt_or_messages: Sequence[Dict[str, str]] | str,
    *,
    user_prompt: Optional[str] = None,
    decoding: Optional[Dict[str, Any]] = None,
    **legacy_kwargs: Any,
) -> GenerationResult:
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

    decode_cfg.setdefault("trailer_reserved_tokens", 80)
    decode_cfg.setdefault("stop_on_trailer", True)

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
    "GenerationResult",
    "generate_json_only",
    "load_model_and_tokenizer",
]

