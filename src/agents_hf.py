from __future__ import annotations
import json, re
from typing import Any, Dict, List, Optional
from .model_loader import generate_json_only
from .strategies import Strategy
from .sanitize import repair_envelope
from .utils import ALLOWED_PERFORMATIVES
from .pseudocode import augment_system_prompt


_PERFORMATIVE_LIST = ", ".join(ALLOWED_PERFORMATIVES)

JSON_GUIDE = (
    "You are one of two collaborating agents. Respond with a SINGLE JSON object ONLY.\n"
    "Fields:\n"
    "- tag: exactly \"[CONTACT]\" when you need your peer, or \"[SOLVED]\" when you are done.\n"
    "- status: one of WORKING, NEED_PEER, PROPOSED, READY_TO_SOLVE, SOLVED.\n"
    "- content.acl: coordination message formatted as 'INTENT: message => next_action'.\n"
    f"  Allowed INTENT values: {_PERFORMATIVE_LIST}.\n"
    "- final_solution: include ONLY when tag is [SOLVED], with key canonical_text.\n"
    "Return ONLY the JSON object. No preamble, no backticks, no extra text.\n"
    "GOOD acl example: 'PROPOSE: outline solution steps => WAIT_FOR_PEER'.\n"
    "BAD acl example: 'I think we should do X' (missing INTENT prefix).\n"
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
        self.system_prompt = augment_system_prompt(system_prompt)
        self.tokenizer = tokenizer
        self.model = model
        self.strategy = strategy

    def _messages(self, task: str, transcript: List[Dict[str, Any]]):
        sys = self.system_prompt
        if self.strategy.prompt_snippet:
            sys = sys + "\n\n" + self.strategy.prompt_snippet
        if self.strategy.json_only:
            sys = sys + "\n\n" + JSON_GUIDE
    def _messages(
        self,
        task: str,
        transcript: List[Dict[str, Any]],
        preparation: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        prep = preparation or {}
        sys_parts: List[str] = []
        if prep.get("system_prefix"):
            sys_parts.append(str(prep["system_prefix"]))
        sys_parts.append(self.system_prompt)
        if self.strategy.json_only and not prep.get("omit_json_guide"):
            sys_parts.append(JSON_GUIDE)
        if prep.get("system_suffix"):
            sys_parts.append(str(prep["system_suffix"]))
        if prep.get("format_hint"):
            sys_parts.append(str(prep["format_hint"]))
        if prep.get("grammar"):
            sys_parts.append("Grammar:\n" + str(prep["grammar"]))
        system_prompt = "\n\n".join([s for s in sys_parts if s])

        peer_context = "{}"
        if transcript:
            last = transcript[-1]
            peer_context = json.dumps(last.get("envelope", {}), ensure_ascii=False)
        usr = f"Task: {task}\nPeer context: {peer_context}\nReturn ONLY the JSON object per schema."
        usr = self.strategy.decorate_prompts(usr, {"agent": self.name})
        return [{"role":"system","content":sys}, {"role":"user","content":usr}]

    def step(self, task: str, transcript: List[Dict[str, Any]]):
        msgs = self._messages(task, transcript)
        decoding = self.strategy.decoding or {}
        raw = generate_json_only(
            self.tokenizer,
            self.model,
            msgs,
            max_new_tokens=decoding.get("max_new_tokens", 256),
            temperature=decoding.get("temperature", 0.0),
            do_sample=decoding.get("do_sample"),
            top_p=decoding.get("top_p"),
            top_k=decoding.get("top_k"),
        )
        user_parts: List[str] = []
        if prep.get("user_prefix"):
            user_parts.append(str(prep["user_prefix"]))
        base = f"Task: {task}\nPeer context: {peer_context}\nReturn ONLY the JSON object per schema."
        user_parts.append(base)
        if prep.get("user_suffix"):
            user_parts.append(str(prep["user_suffix"]))
        if prep.get("extra_user_instructions"):
            user_parts.append(str(prep["extra_user_instructions"]))
        user_prompt = "\n\n".join([p for p in user_parts if p])
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    def step(
        self,
        task: str,
        transcript: List[Dict[str, Any]],
        preparation: Optional[Dict[str, Any]] = None,
    ):
        msgs = self._messages(task, transcript, preparation)
        decoding = dict(self.strategy.decoding or {})
        if preparation and preparation.get("decoding_override"):
            decoding.update(preparation["decoding_override"])
        raw = generate_json_only(self.tokenizer, self.model, msgs, decoding=decoding)
        env = _extract_json(raw) or {"status": "WORKING", "tag": "[CONTACT]", "content": {"note": "fallback"}}
        env = repair_envelope(env)
        return env, raw

