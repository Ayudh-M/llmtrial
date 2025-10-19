from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

TINY_REPO = "roneneldan/TinyStories-1M"
TINY_TOKENIZER = "EleutherAI/gpt-neo-125M"

def _resolve_dtype(dtype: str | None):
    if not dtype:
        return None
    d = dtype.lower()
    if d in ("bfloat16","bf16"): return torch.bfloat16
    if d in ("float16","fp16"):  return torch.float16
    if d in ("float32","fp32"):  return torch.float32
    return None

def load_model_and_tokenizer(repo_id: str, dtype=None, quant=None):
    tok = AutoTokenizer.from_pretrained(repo_id)
    kwargs = {}
    if quant == "4bit":
        # requires: pip install bitsandbytes accelerate
        kwargs.update(
            device_map="auto",
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quant == "8bit":
        kwargs.update(device_map="auto", load_in_8bit=True)
    else:
        kwargs.update(device_map="auto")  # fp16/bf16 if your GPU can take it
    model = AutoModelForCausalLM.from_pretrained(repo_id, **kwargs)
    model.eval()
    return tok, model

def _has_chat_template(tokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))

def build_inputs(tokenizer, messages_or_text, add_generation_prompt: bool = True):
    if _has_chat_template(tokenizer):
        assert isinstance(messages_or_text, list), "Chat template path requires messages list"
        return tokenizer.apply_chat_template(messages_or_text, return_tensors="pt", add_generation_prompt=add_generation_prompt)
    if isinstance(messages_or_text, list):
        flat = []
        for m in messages_or_text:
            role = m.get("role","user")
            content = m.get("content","")
            flat.append(f"{role.upper()}: {content}")
        prompt = "\n".join(flat) + "\nASSISTANT:"
    else:
        prompt = str(messages_or_text)
    return tokenizer(prompt, return_tensors="pt").input_ids

@torch.inference_mode()
def _generate(
    tokenizer,
    model,
    messages,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    do_sample: Optional[bool] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
):
    if do_sample is None:
        do_sample = bool(temperature and float(temperature) > 0.0)
    input_ids = build_inputs(tokenizer, messages, add_generation_prompt=True)
    input_ids = input_ids.to(model.device)
    gen_kwargs: Dict[str, Any] = {
        "do_sample": do_sample,
        "temperature": float(temperature or 0.0),
        "max_new_tokens": int(max_new_tokens or 256),
    }
    if top_p is not None:
        gen_kwargs["top_p"] = float(top_p)
    if top_k is not None:
        gen_kwargs["top_k"] = int(top_k)
    out = model.generate(
        input_ids=input_ids,
        **gen_kwargs,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def generate_json_only(
    tokenizer,
    model,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    do_sample: Optional[bool] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> str:
    return _generate(
        tokenizer,
        model,
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
    )
