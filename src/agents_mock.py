from __future__ import annotations

"""Deterministic agents used by tests and offline simulations."""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .strategies import Strategy, build_strategy


@dataclass
class MockAgent:
    name: str
    answer: str
    strategy: Strategy | None = None
    persona: str = "assistant"

    def __post_init__(self) -> None:
        if self.strategy is None:
            self.strategy = build_strategy("json_schema")
        self._turn = 0

    def _json_envelope(self, solved: bool) -> Dict[str, Any]:
        status = "SOLVED" if solved else "WORKING"
        tag = "[SOLVED]" if solved else "[CONTACT]"
        public = "[SOLVED] Solution prepared" if solved else "[CONTACT] Continuing"
        content: Dict[str, Any]
        if self.strategy.metadata.get("requires_acl"):
            if solved:
                acl = f"SOLVED: {self.answer} => CONFIRM"
            else:
                acl = f"PROPOSE: outline approach for {self.answer}"
            content = {"acl": acl}
        else:
            content = {"summary": self.answer if solved else "working"}
        envelope: Dict[str, Any] = {
            "role": self.persona,
            "domain": "general",
            "task_understanding": "Understood",
            "public_message": public,
            "artifact": {"type": "results", "content": {"answer": self.answer if solved else None}},
            "needs_from_peer": [],
            "handoff_to": "peer",
            "status": status,
            "tag": tag,
            "content": content,
            "meta": {"strategy_id": self.strategy.id},
        }
        if solved:
            envelope["final_solution"] = {"canonical_text": self.answer}
        return envelope

    def step(self, task: str, transcript: List[Dict[str, Any]]) -> Tuple[Any, str]:
        self._turn += 1
        solved = self._turn > 1 or not self.strategy.json_only

        if self.strategy.json_only:
            envelope = self._json_envelope(solved)
            raw = json.dumps(envelope, ensure_ascii=False)
            return envelope, raw

        # Text-only strategies
        if self.strategy.metadata.get("requires_dsl"):
            text = f"PLAN: {self.answer} => EXECUTE"
        else:
            text = f"Answer: {self.answer}" if solved else "Considering..."
        return text, text
