import pytest

pytestmark = pytest.mark.skip(
    reason="Legacy control-trailer/consensus disabled in simplified fixed-turn runner"
)

from src.agents_mock import MockAgent
from src.controller import run_controller
from src.strategies import build_strategy, list_strategy_ids
from src.template_loader import get_scenario, load_registry


REGISTRY = load_registry()
SCENARIO_IDS = tuple(sorted(REGISTRY["scenarios"].keys()))


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
@pytest.mark.parametrize("strategy_id", list_strategy_ids())
def test_mock_agents_compatible_with_all_strategies(scenario_id: str, strategy_id: str) -> None:
    scenario = get_scenario(scenario_id)
    task = str(scenario.get("task", "Return TRUE")).strip() or "Return TRUE"
    strategy = build_strategy(strategy_id)
    agent_a = MockAgent("A", "TRUE", strategy=strategy)
    agent_b = MockAgent("B", "TRUE", strategy=strategy)

    result = run_controller(
        task,
        agent_a,
        agent_b,
        max_rounds=strategy.max_rounds,
        strategy=strategy,
    )

    assert result["status"] == "CONSENSUS"
