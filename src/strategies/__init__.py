from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple, Union

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
    cfg: Union[str, StrategyDefinition, Strategy, Mapping[str, Any]],
    *,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Strategy:
    if isinstance(cfg, Strategy):
        return cfg

    merged_overrides: Dict[str, Any] = dict(overrides or {})
    if isinstance(cfg, StrategyDefinition):
        definition = cfg
    elif isinstance(cfg, str):
        definition = get_strategy_definition(cfg)
    elif isinstance(cfg, Mapping):
        raw = dict(cfg)
        strategy_id = raw.get("id") or raw.get("name") or raw.get("strategy")
        if strategy_id and strategy_id in STRATEGY_REGISTRY:
            definition = get_strategy_definition(str(strategy_id))
            config_overrides = {
                k: v for k, v in raw.items() if k not in {"id", "name", "strategy"}
            }
            if "validator" in config_overrides and "validator_id" not in config_overrides:
                config_overrides["validator_id"] = config_overrides.pop("validator")
            if "prompt_snippet" in config_overrides:
                meta = dict(definition.metadata)
                if "metadata" in config_overrides and isinstance(config_overrides["metadata"], Mapping):
                    meta.update(dict(config_overrides["metadata"]))
                meta["prompt_snippet"] = config_overrides.pop("prompt_snippet")
                config_overrides["metadata"] = meta
            merged_overrides = {**config_overrides, **merged_overrides}
        else:
            data = dict(raw)
            if merged_overrides:
                # Merge decoding dictionaries rather than replacing.
                if "decoding" in merged_overrides and "decoding" in data:
                    merged = dict(data["decoding"])
                    merged.update(dict(merged_overrides["decoding"]))
                    data["decoding"] = merged
                data.update({k: v for k, v in merged_overrides.items() if k != "decoding"})
            if strategy_id:
                data.setdefault("id", str(strategy_id))
                data.setdefault("name", str(strategy_id))
            else:
                data.setdefault("id", "custom")
                data.setdefault("name", "custom")
            decoding = data.get("decoding")
            if decoding is not None:
                data["decoding"] = dict(decoding)
            metadata = data.get("metadata") or {}
            data["metadata"] = dict(metadata)
            validator_params = data.get("validator_params")
            if validator_params is not None:
                data["validator_params"] = dict(validator_params)
            if "validator" in data and "validator_id" not in data:
                data["validator_id"] = data.pop("validator")
            prompt_snippet = data.pop("prompt_snippet", None)
            if prompt_snippet is not None:
                data.setdefault("metadata", {})
                data["metadata"].setdefault("prompt_snippet", prompt_snippet)
            return Strategy(**data)
    else:
        raise TypeError(
            "Strategy configuration must be a strategy id, definition, mapping, or Strategy instance."
        )

    if "decoding" in merged_overrides and merged_overrides["decoding"] is not None:
        merged = dict(definition.decoding)
        merged.update(merged_overrides["decoding"])
        merged_overrides["decoding"] = merged

    return definition.instantiate(**merged_overrides)


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
