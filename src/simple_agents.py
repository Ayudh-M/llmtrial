# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def _format_prompt(system_prompt: str, incoming: str) -> str:
    return (
        f"<<SYSTEM>>\n{system_prompt}\n<</SYSTEM>>\n\n"
        f"<<PEER>>\n{incoming}\n<</PEER>>\n\n"
        f"<<YOU>>\n"
    )


@dataclass
class GenConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.05


class SimpleHF:
    def __init__(self, model_id: str, device: str | None = None):
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def respond(
        self, system_prompt: str, incoming: str, cfg: GenConfig
    ) -> Tuple[str, int, int, str]:
        prompt = _format_prompt(system_prompt, incoming)
        enc = self.tok(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        gen_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=cfg.do_sample,
            repetition_penalty=cfg.repetition_penalty,
            pad_token_id=
            self.tok.eos_token_id
            if self.tok.pad_token_id is None
            else self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id,
        )
        new_ids = gen_ids[0, input_ids.shape[1]:]
        text = self.tok.decode(new_ids, skip_special_tokens=True).strip()
        stop = "length" if new_ids.shape[0] >= cfg.max_new_tokens else "eos_or_sample"
        return text, int(input_ids.numel()), int(new_ids.numel()), stop


def seed_everything(seed: int | None):
    if seed is not None:
        set_seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
