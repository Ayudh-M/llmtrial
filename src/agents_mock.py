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
            env = {
                "tag": "[CONTACT]",
                "status": "PROPOSED",
                "content": {
                    "acl": f"PROPOSE: sharing candidate '{self.solution_text}' => WAIT_FOR_PEER",
                    "proposal": self.solution_text,
                },
            }
            return env, str(env)
        env = {
            "tag": "[SOLVED]",
            "status": "SOLVED",
            "content": {
                "acl": "SOLVED: submitting final answer => END_DIALOGUE",
            },
            "final_solution": {"canonical_text": self.solution_text},
        }
        return env, str(env)
