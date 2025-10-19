from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from .dsl import DSLSpec, default_dsl_spec

@dataclass
class Strategy:
    name: str
    json_only: bool = True
    allow_cot: bool = False
    max_rounds: int = 8
    decoding: Optional[Dict[str, Any]] = None
    consensus_mode: str = "review_handshake"
    dsl_spec: DSLSpec = field(default_factory=default_dsl_spec)

def build_strategy(cfg: Dict[str, Any]) -> Strategy:
    dsl_cfg = cfg.get("dsl") if isinstance(cfg, dict) else None
    spec = default_dsl_spec()
    if isinstance(dsl_cfg, dict):
        grammar = dsl_cfg.get("grammar", spec.grammar)
        keywords = dsl_cfg.get("keywords", spec.keywords)
        artifact_types = dsl_cfg.get("artifact_types", spec.artifact_types)
        allowed_status = dsl_cfg.get("allowed_status", spec.allowed_status)
        allowed_tags = dsl_cfg.get("allowed_tags", spec.allowed_tags)
        content_rules = dsl_cfg.get("artifact_content_rules", spec.artifact_content_rules)
        spec = DSLSpec(
            grammar=str(grammar),
            keywords=list(keywords),
            artifact_types=list(artifact_types),
            allowed_status=list(allowed_status),
            allowed_tags=list(allowed_tags),
            artifact_content_rules=dict(content_rules or {}),
        )
    return Strategy(
        name=cfg.get("id") or cfg.get("name", "S1"),
        json_only=True if cfg.get("json_envelope_schema") else cfg.get("json_only", True),
        allow_cot=cfg.get("allow_cot", False),
        max_rounds=cfg.get("max_rounds", 8),
        decoding=cfg.get("decoding") or {"do_sample": False, "temperature": 0.0, "max_new_tokens": 256},
        consensus_mode=cfg.get("consensus_mode", "review_handshake"),
        dsl_spec=spec,
    )
