from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class Strategy:
    name: str
    json_only: bool = True
    allow_cot: bool = False
    max_rounds: int = 8
    decoding: Optional[Dict[str, Any]] = None
    consensus_mode: str = "review_handshake"
    prompt_snippet: Optional[str] = None
    validator_id: Optional[str] = None
    validator_params: Optional[Dict[str, Any]] = None
    envelope_required: bool = True

def build_strategy(cfg: Dict[str, Any]) -> Strategy:
    return Strategy(
        name=cfg.get("id") or cfg.get("name", "S1"),
        json_only=True if cfg.get("json_envelope_schema") else cfg.get("json_only", True),
        allow_cot=cfg.get("allow_cot", False),
        max_rounds=cfg.get("max_rounds", 8),
        decoding=cfg.get("decoding") or {"do_sample": False, "temperature": 0.0, "max_new_tokens": 256},
        consensus_mode=cfg.get("consensus_mode", "review_handshake"),
        prompt_snippet=cfg.get("prompt_snippet"),
        validator_id=cfg.get("validator"),
        validator_params=cfg.get("validator_params"),
        envelope_required=cfg.get("envelope_required", True),
    )
