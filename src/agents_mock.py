from __future__ import annotations
from typing import Any, Dict, List, Tuple

from .utils import sha256_hex


class MockAgent:
    def __init__(
        self,
        name: str,
        solution_text: str,
        *,
        role: str | None = None,
        domain: str = "mock-domain",
        handoff_to: str = "peer",
        artifact_type: str = "results",
    ):
        self.name = name
        self.role = role or name
        self.domain = domain
        self.handoff_to = handoff_to
        self.artifact_type = artifact_type
        self.solution_text = solution_text
        self._has_proposed = False

    def _base_envelope(self, task: str) -> Dict[str, Any]:
        task_understanding = task.strip() or "mock task"
        return {
            "role": self.role,
            "domain": self.domain,
            "task_understanding": task_understanding,
            "artifact": {"type": self.artifact_type, "content": {}},
            "needs_from_peer": [],
            "handoff_to": self.handoff_to,
            "content": {},
        }

    def step(self, task: str, transcript: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
        env = self._base_envelope(task)
        if not self._has_proposed:
            self._has_proposed = True
            env.update(
                {
                    "tag": "[CONTACT]",
                    "status": "PROPOSED",
                    "public_message": "[CONTACT] proposing candidate",
                }
            )
            env["artifact"]["content"] = {"proposal": self.solution_text}
            return env, str(env)

        env.update(
            {
                "tag": "[SOLVED]",
                "status": "SOLVED",
                "public_message": "[SOLVED] consensus candidate",
                "content": {"verdict": "ACCEPT"},
                "final_solution": {
                    "canonical_text": self.solution_text,
                    "sha256": sha256_hex(self.solution_text),
                },
            }
        )
        env["artifact"]["content"] = {"final": self.solution_text}
        return env, str(env)
