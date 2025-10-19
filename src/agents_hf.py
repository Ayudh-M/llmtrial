from __future__ import annotations
import json, re
from typing import Any, Dict, List, Optional
from .model_loader import generate_json_only
from .strategies import Strategy
from .sanitize import repair_envelope


JSON_GUIDE = (
    "You are one of two collaborating agents. Respond with a SINGLE JSON object ONLY.\n"
    "Fields:\n"
    "- tag: exactly \"[CONTACT]\" when you need your peer, or \"[SOLVED]\" when you are done.\n"
    "- status: one of WORKING, NEED_PEER, PROPOSED, READY_TO_SOLVE, SOLVED.\n"
    "- content: an object for coordination when tag is [CONTACT].\n"
    "- final_solution: include this object ONLY when tag is [SOLVED], with key canonical_text.\n"
    "Return ONLY the JSON object. No preamble, no backticks, no extra text.\n"
    "Example shape (values are placeholders):\n"
    "{\n"
    "  \"tag\": \"[CONTACT]\",\n"
    "  \"status\": \"PROPOSED\",\n"
    "  \"content\": {\"note\": \"...\"}\n"
    "}\n"
)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

class HFChatAgent:
    def __init__(self, name: str, system_prompt: str, tokenizer, model, strategy: Strategy):
        self.name = name
        self.system_prompt = system_prompt
        self.tokenizer = tokenizer
        self.model = model
        self.strategy = strategy

    def _messages(self, task: str, transcript: List[Dict[str, Any]]):
        sys = self.system_prompt + "\n\n" + JSON_GUIDE if self.strategy.json_only else self.system_prompt
        peer_context = "{}"
        if transcript:
            last = transcript[-1]
            peer_context = json.dumps(last.get("envelope", {}), ensure_ascii=False)
        usr = f"Task: {task}\nPeer context: {peer_context}\nReturn ONLY the JSON object per schema."
        return [{"role":"system","content":sys}, {"role":"user","content":usr}]

    def step(self, task: str, transcript: List[Dict[str, Any]]):
        msgs = self._messages(task, transcript)
        raw = generate_json_only(
            self.tokenizer, self.model, msgs,
            max_new_tokens=(self.strategy.decoding or {}).get("max_new_tokens", 256),
            temperature=(self.strategy.decoding or {}).get("temperature", 0.0)
        )
        env = _extract_json(raw) or {"status": "WORKING", "tag": "[CONTACT]", "content": {"note": "fallback"}}
        env = repair_envelope(env)
        return env, raw

