from __future__ import annotations
import yaml, json
from pathlib import Path
from typing import Any, Dict

from .strategies import StrategyDefinition, get_strategy_definition

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "prompts" / "registry.yaml"


def _scenario_lookup_id(scenario_id: str) -> str:
    parts = str(scenario_id).split(":")
    if parts and parts[-1].startswith("rep="):
        parts = parts[:-1]
    return ":".join(parts)

def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data

def load_registry() -> Dict[str, Any]:
    if not REGISTRY.exists():
        raise FileNotFoundError(f"Registry not found: {REGISTRY}")
    data = _load_yaml(REGISTRY)
    if "scenarios" not in data or not isinstance(data["scenarios"], dict):
        raise ValueError("Registry must contain a 'scenarios' mapping.")
    return data

def load_strategy(name: str) -> StrategyDefinition:
    try:
        return get_strategy_definition(name)
    except KeyError as exc:
        raise KeyError(f"Strategy id not found in registry: {name}") from exc

def load_roleset(path_str: str) -> Dict[str, Any]:
    path = (ROOT / path_str).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Roleset not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            return json.load(f)
        return yaml.safe_load(f)

def get_scenario(scenario_id: str) -> Dict[str, Any]:
    reg = load_registry()
    lookup_id = _scenario_lookup_id(scenario_id)
    try:
        return reg["scenarios"][lookup_id]
    except KeyError as exc:
        raise KeyError(f"Scenario id not found in registry: {lookup_id}") from exc
