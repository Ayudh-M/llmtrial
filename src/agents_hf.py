from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .model_loader import generate_json_only
from .pseudocode import augment_system_prompt
from .sanitize import repair_envelope
from .strategies import Strategy
from .utils import ALLOWED_PERFORMATIVES


_PERFORMATIVE_LIST = ", ".join(ALLOWED_PERFORMATIVES)

JSON_GUIDE = (
    "You are one of two collaborating agents. Respond with a SINGLE JSON object ONLY.\n"
    "Fields:\n"
    "- tag: exactly '[CONTACT]' when you need your peer, or '[SOLVED]' when you are done.\n"
    "- status: one of WORKING, NEED_PEER, PROPOSED, READY_TO_SOLVE, SOLVED.\n"
    "- content.acl: coordination message formatted as 'INTENT: message => next_action'.\n"
    f"  Allowed INTENT values: {_PERFORMATIVE_LIST}.\n"
    "- final_solution: include when you share a candidate resolution or confirm acceptance. canonical_text must be a short, stable string that your partner can copy exactly.\n"
    "Consensus handshake: when you accept your partner's final solution, reply with tag '[SOLVED]' and status 'SOLVED', set content.verdict to 'ACCEPT', and set final_solution.canonical_text to EXACTLY match the partner's canonical_text. If you cannot accept, respond with status 'REVISED' explaining what is missing.\n"
    "Return ONLY the JSON object. No preamble, no backticks, no extra text.\n"
    "GOOD acl example: 'PROPOSE: outline solution steps => WAIT_FOR_PEER'.\n"
    "BAD acl example: 'I think we should do X' (missing INTENT prefix).\n"
)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _maybe_add_snippet(strategy: Strategy) -> Optional[str]:
    meta = strategy.metadata or {}
    snippet = meta.get("prompt_snippet")
    if not snippet:
        return None
    return str(snippet)


class HFChatAgent:
    """Wrapper that turns a HF causal LM into a controller-compatible agent."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tokenizer,
        model,
        strategy: Strategy,
    ) -> None:
        self.name = name
        self.base_system_prompt = augment_system_prompt(system_prompt)
        self.tokenizer = tokenizer
        self.model = model
        self.strategy = strategy

    # -- prompt assembly -------------------------------------------------
    def _system_prompt(self, preparation: Optional[Mapping[str, Any]]) -> str:
        prep = preparation or {}
        parts: List[str] = []
        if prep.get("system_prefix"):
            parts.append(str(prep["system_prefix"]))
        parts.append(self.base_system_prompt)

        snippet = _maybe_add_snippet(self.strategy)
        if snippet:
            parts.append(snippet)

        if self.strategy.json_only and not prep.get("omit_json_guide"):
            parts.append(JSON_GUIDE)

        if prep.get("system_suffix"):
            parts.append(str(prep["system_suffix"]))

        return "\n\n".join([p for p in parts if p])

    def _user_prompt(
        self,
        task: str,
        transcript: List[Dict[str, Any]],
        preparation: Optional[Mapping[str, Any]],
    ) -> str:
        prep = preparation or {}
        peer_context = "{}"
        if transcript:
            peer_context = json.dumps(transcript[-1].get("envelope", {}), ensure_ascii=False)

        parts: List[str] = []
        if prep.get("user_prefix"):
            parts.append(str(prep["user_prefix"]))

        base = f"Task: {task}\nPeer context: {peer_context}\nReturn ONLY the JSON object per schema."
        parts.append(self.strategy.decorate_prompts(base, {"agent": self.name}))

        if prep.get("user_suffix"):
            parts.append(str(prep["user_suffix"]))
        if prep.get("extra_user_instructions"):
            parts.append(str(prep["extra_user_instructions"]))

        return "\n\n".join([p for p in parts if p])

    def _messages(
        self,
        task: str,
        transcript: List[Dict[str, Any]],
        preparation: Optional[Mapping[str, Any]],
    ) -> List[Dict[str, str]]:
        system_prompt = self._system_prompt(preparation)
        user_prompt = self._user_prompt(task, transcript, preparation)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    # -- controller hook -------------------------------------------------
    def step(
        self,
        task: str,
        transcript: List[Dict[str, Any]],
        preparation: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], str]:
        messages = self._messages(task, transcript, preparation)
        decoding = dict(self.strategy.decoding or {})
        if preparation and preparation.get("decoding_override"):
            decoding.update(preparation["decoding_override"])  # type: ignore[arg-type]

        raw = generate_json_only(self.tokenizer, self.model, messages, decoding=decoding)
        envelope = _extract_json(raw)
        if envelope is None:
            envelope = {"status": "WORKING", "tag": "[CONTACT]", "content": {"note": "fallback"}}

        envelope = repair_envelope(envelope)
        return envelope, raw


__all__ = ["HFChatAgent", "JSON_GUIDE"]

