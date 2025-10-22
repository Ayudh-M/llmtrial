import json
from pathlib import Path

import yaml

from src.strategies import STRATEGY_REGISTRY

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "prompts" / "registry.yaml"

EXPECTED_SCENARIOS = {
    "llm_eval_preference",
    "coding_bug_fix_planner",
    "coding_bug_fix_reflexion",
    "moderation_constitutional",
    "moderation_boundary_design",
    "grounding_science_summary",
    "grounding_peer_review",
    "reasoning_plan_solve",
    "reasoning_reflexion",
}


def load_registry() -> dict:
    with REGISTRY_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_registry_contains_expected_scenarios() -> None:
    data = load_registry()
    scenarios = data.get("scenarios", {})
    assert EXPECTED_SCENARIOS.issubset(set(scenarios.keys()))



def test_rolesets_exist_and_have_minimal_shape() -> None:
    data = load_registry()
    scenarios = data["scenarios"]
    for name, config in scenarios.items():
        roleset_rel = config["roleset"]
        roleset_path = ROOT / roleset_rel
        assert roleset_path.exists(), f"Missing roleset for {name}: {roleset_rel}"
        with roleset_path.open("r", encoding="utf-8") as f:
            blob = json.load(f)
        assert "agent_a" in blob and "agent_b" in blob
        for label in ("agent_a", "agent_b"):
            agent = blob[label]
            assert agent.get("name"), f"{name} {label} missing name"
            system = agent.get("system", "")
            assert "JSON" in system, f"{name} {label} system prompt must mention JSON"
        meta = blob.get("meta", {})
        assert "task_kind" in meta



def test_scenarios_reference_known_strategies() -> None:
    data = load_registry()
    scenarios = data["scenarios"]
    for name, config in scenarios.items():
        strategy_id = config["strategy"]
        assert strategy_id in STRATEGY_REGISTRY, f"Unknown strategy {strategy_id} in {name}"
