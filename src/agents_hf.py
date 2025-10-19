from __future__ import annotations

"""Hugging Face powered chat agent respecting strategy configuration."""

import json
from typing import Any, Dict, List, Optional

from .model_loader import generate_chat_completion, generate_json_only
from .strategies import Strategy
from .json_enforcer import coerce_minimal_defaults


class HFChatAgent:
    def __init__(self, name: str, system_prompt: str, tokenizer, model, strategy: Strategy):
        self.name = name
        self.system_prompt = system_prompt.strip()
        self.tokenizer = tokenizer
        self.model = model
        self.strategy = strategy

    def _build_messages(
        self,
        task: str,
        transcript: List[Dict[str, Any]],
        *,
        preparation: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        prep = preparation or {}
        system_parts = [self.system_prompt]
        if prep.get("system_suffix"):
            system_parts.append(str(prep["system_suffix"]))
        system_text = "\n\n".join([part for part in system_parts if part])

        peer_context = transcript[-1]["raw"] if transcript else "(no previous message)"
        user_parts = [f"Task: {task}"]
        user_parts.append(f"Peer context: {peer_context}")
        if prep.get("user_suffix"):
            user_parts.append(str(prep["user_suffix"]))
        user_prompt = "\n\n".join(user_parts)
        user_prompt = self.strategy.decorate_prompts(user_prompt, {"agent": self.name})

        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_prompt},
        ]

    def step(
        self,
        task: str,
        transcript: List[Dict[str, Any]],
        *,
        preparation: Optional[Dict[str, Any]] = None,
    ):
        messages = self._build_messages(task, transcript, preparation=preparation)
        decoding = dict(self.strategy.decoding or {})

        if self.strategy.json_only:
            raw = generate_json_only(self.tokenizer, self.model, messages, decoding=decoding)
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = coerce_minimal_defaults({})
            return payload, raw

        raw = generate_chat_completion(self.tokenizer, self.model, messages, decoding=decoding)
        return raw.strip(), raw
