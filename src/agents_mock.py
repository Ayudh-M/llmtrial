from __future__ import annotations
from typing import Any, Dict, List, Tuple

class MockAgent:
    def __init__(self, name: str, solution_text: str):
        self.name = name
        self.solution_text = solution_text
        self._has_proposed = False

    def step(self, task: str, transcript: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
        if not self._has_proposed:
            self._has_proposed = True
            env = {"tag":"[CONTACT]", "status":"PROPOSED", "content":{"proposal": self.solution_text}}
            return env, str(env)
        env = {"tag":"[SOLVED]", "status":"SOLVED", "final_solution":{"canonical_text": self.solution_text}}
        return env, str(env)


class ConciseTextAgent:
    """Mock agent that speaks in short text snippets and optional final answer."""

    def __init__(self, name: str, messages: List[str], final_answer: str):
        self.name = name
        self._messages = list(messages)
        self._final_answer = final_answer
        self._turn = 0

    def step(self, task: str, transcript: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
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
