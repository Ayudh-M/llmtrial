from __future__ import annotations

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
    except KeyError as exc:
        raise KeyError(f"Unknown strategy '{strategy_id}'.") from exc


def build_strategy(
    cfg: Union[str, StrategyDefinition, Strategy, Mapping[str, Any]]
) -> Strategy:
    if isinstance(cfg, Strategy):
        return cfg

    overrides: Dict[str, Any] = {}
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
        overrides = {
            k: v for k, v in dict(cfg).items() if k not in {"id", "name", "strategy"}
        }
    else:
        raise TypeError(
            "Strategy configuration must be a strategy id, definition, mapping, or Strategy instance."
        )

    if "decoding" in overrides and overrides["decoding"] is not None:
        merged = dict(definition.decoding)
        merged.update(overrides["decoding"])
        overrides["decoding"] = merged

    return definition.instantiate(**overrides)


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
