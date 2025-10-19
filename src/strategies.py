from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

from jsonschema import Draft7Validator

from .schemas import get_envelope_validator

@dataclass
class Strategy:
    name: str
    json_only: bool = True
    allow_cot: bool = False
    max_rounds: int = 8
    decoding: Dict[str, Any] | None = None
    consensus_mode: str = "review_handshake"
    json_envelope_schema: Optional[str] = None
    envelope_validator: Optional[Draft7Validator] = None

_DEFAULT_DECODING = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 256}


def build_strategy(cfg: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Strategy:
    overrides = overrides or {}
    decoding = dict(_DEFAULT_DECODING)
    decoding.update(cfg.get("decoding") or {})
    decoding.update(overrides.get("decoding") or {})

    schema_ref = overrides.get("json_envelope_schema") or cfg.get("json_envelope_schema")
    validator: Optional[Draft7Validator] = get_envelope_validator(schema_ref) if schema_ref else None

    strategy = Strategy(
        name=cfg.get("id") or cfg.get("name", "S1"),
        json_only=True if schema_ref else cfg.get("json_only", True),
        allow_cot=cfg.get("allow_cot", False),
        max_rounds=overrides.get("max_rounds", cfg.get("max_rounds", 8)),
        decoding=decoding,
        consensus_mode=overrides.get("consensus_mode", cfg.get("consensus_mode", "review_handshake")),
        json_envelope_schema=schema_ref,
        envelope_validator=validator,
    )
    return strategy
