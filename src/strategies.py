from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

from .schemas import Envelope

@dataclass
class Strategy:
    name: str
    json_only: bool = True
    allow_cot: bool = False
    max_rounds: int = 8
    decoding: Optional[Dict[str, Any]] = None
    consensus_mode: str = "review_handshake"
    output_format_hint: Optional[str] = None
    grammar: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def prepare_prompt(
        self,
        task: str,
        transcript: List[Dict[str, Any]],
        *,
        actor: str,
        agent_name: str,
    ) -> Dict[str, Any]:
        """Return additional prompt shaping hints for the agent."""
        hints: Dict[str, Any] = {}
        if self.output_format_hint:
            hints["format_hint"] = self.output_format_hint
        if self.grammar:
            hints["grammar"] = self.grammar
        return hints

    def validate_message(
        self,
        envelope: Envelope,
        *,
        raw: str,
        original: Dict[str, Any],
        transcript: List[Dict[str, Any]],
        actor: str,
        agent_name: str,
    ) -> Dict[str, Any]:
        """Inspect an agent message and return validation metadata."""
        return {
            "ok": True,
            "tag": envelope.tag,
            "status": envelope.status,
        }

    def postprocess(
        self,
        envelope: Envelope,
        *,
        raw: str,
        validation: Dict[str, Any],
        transcript: List[Dict[str, Any]],
        actor: str,
        agent_name: str,
    ) -> Tuple[Envelope | Dict[str, Any], Dict[str, Any]]:
        """Allow strategies to amend envelopes or attach parsed structures."""
        return envelope, {"parsed_envelope": envelope.model_dump()}

    def should_stop(
        self,
        envelope: Envelope,
        *,
        validation: Dict[str, Any],
        transcript: List[Dict[str, Any]],
        actor: str,
        agent_name: str,
    ) -> Tuple[bool, Optional[str]]:
        """Return a flag (and optional reason) indicating whether to stop early."""
        return False, None

def build_strategy(cfg: Dict[str, Any]) -> Strategy:
    return Strategy(
        name=cfg.get("id") or cfg.get("name", "S1"),
        json_only=True if cfg.get("json_envelope_schema") else cfg.get("json_only", True),
        allow_cot=cfg.get("allow_cot", False),
        max_rounds=cfg.get("max_rounds", 8),
        decoding=cfg.get("decoding") or {"do_sample": False, "temperature": 0.0, "max_new_tokens": 256},
        consensus_mode=cfg.get("consensus_mode", "review_handshake"),
        output_format_hint=cfg.get("output_format_hint"),
        grammar=cfg.get("dsl_grammar") or cfg.get("grammar"),
        metadata=cfg.get("metadata", {}),
    )
