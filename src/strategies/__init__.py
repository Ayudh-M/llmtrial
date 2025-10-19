from __future__ import annotations

"""Convenience helpers for interacting with the strategy registry."""

from typing import Any, Dict, Mapping, Tuple, Union

from .registry import (
    AgentProfile,
    Strategy,
    StrategyDefinition,
    STRATEGY_REGISTRY,
)

REGISTRY = STRATEGY_REGISTRY


def list_strategy_ids() -> Tuple[str, ...]:
    return tuple(sorted(STRATEGY_REGISTRY.keys()))


def get_strategy_definition(strategy_id: str) -> StrategyDefinition:
    try:
        return STRATEGY_REGISTRY[strategy_id]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise KeyError(f"Unknown strategy '{strategy_id}'.") from exc


def build_strategy(
    cfg: Union[str, StrategyDefinition, Strategy, Mapping[str, Any]],
    *,
    overrides: Mapping[str, Any] | None = None,
) -> Strategy:
    if isinstance(cfg, Strategy):
        base_definition = STRATEGY_REGISTRY.get(cfg.id)
        if base_definition is None:
            return cfg
        return base_definition.instantiate()

    overrides_dict: Dict[str, Any] = dict(overrides or {})

    if isinstance(cfg, StrategyDefinition):
        definition = cfg
    elif isinstance(cfg, str):
        definition = get_strategy_definition(cfg)
    elif isinstance(cfg, Mapping):
        strategy_id = (
            cfg.get("id")
            or cfg.get("name")
            or cfg.get("strategy")
        )
        if not strategy_id:
            raise ValueError("Strategy configuration requires an 'id'.")
        definition = get_strategy_definition(str(strategy_id))
        for key, value in cfg.items():
            if key not in {"id", "name", "strategy"}:
                overrides_dict.setdefault(key, value)
    else:  # pragma: no cover - defensive
        raise TypeError("Unsupported strategy configuration type.")

    if "decoding" in overrides_dict and overrides_dict["decoding"] is not None:
        merged = dict(definition.decoding)
        merged.update(overrides_dict["decoding"])
        overrides_dict["decoding"] = merged

    return definition.instantiate(**overrides_dict)


__all__ = [
    "AgentProfile",
    "Strategy",
    "StrategyDefinition",
    "STRATEGY_REGISTRY",
    "REGISTRY",
    "list_strategy_ids",
    "get_strategy_definition",
    "build_strategy",
]
