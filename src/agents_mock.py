from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .strategies import Strategy
from .utils import sha256_hex


class MockAgent:
    """Simple deterministic agent for controller tests."""

    def __init__(
        self,
        name: str,
        solution_text: str,
        *,
        strategy: Optional[Strategy] = None,
    ) -> None:
        self.name = name
        self.solution_text = solution_text
        self.strategy = strategy or Strategy(id="mock", name=f"{name}-mock")
        self._has_proposed = False
        self.last_preparation: Optional[Dict[str, Any]] = None

    def step(
        self,
        task: str,
        transcript: List[Dict[str, Any]],
        preparation: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], str]:
        if preparation is not None:
            self.last_preparation = preparation

        if not self._has_proposed:
            self._has_proposed = True
            envelope = {
                "tag": "[CONTACT]",
                "status": "PROPOSED",
                "content": {
                    "acl": f"PROPOSE: sharing candidate '{self.solution_text}' => WAIT_FOR_PEER",
                    "proposal": self.solution_text,
                },
            }
            return envelope, str(envelope)

        envelope = {
            "tag": "[SOLVED]",
            "status": "SOLVED",
            "content": {
                "acl": "SOLVED: submitting final answer => END_DIALOGUE",
                "verdict": "ACCEPT",
            },
            "final_solution": {
                "canonical_text": self.solution_text,
                "sha256": sha256_hex(self.solution_text),
            },
        }
        return envelope, str(envelope)


class ConciseTextAgent:
    """Mock agent that emits short text snippets before a final answer."""

    def __init__(self, name: str, messages: List[str], final_answer: str) -> None:
        self.name = name
        self._messages = list(messages)
        self._final_answer = final_answer
        self._turn = 0

    def step(
        self, task: str, transcript: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], str]:
        if self._turn < len(self._messages):
            text = self._messages[self._turn]
            self._turn += 1
            payload = {"text": text}
            return payload, text

        payload = {
            "text": f" Final: {self._final_answer}. ",
            "final_solution": {"canonical_text": f" {self._final_answer} "},
        }
        self._turn += 1
        return payload, payload["text"]


__all__ = ["MockAgent", "ConciseTextAgent"]
