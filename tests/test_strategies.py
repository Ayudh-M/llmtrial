import pytest

from src.strategies import (
    REGISTRY,
    STRATEGY_REGISTRY,
    build_strategy,
    get_strategy_definition,
    list_strategy_ids,
)


def test_registry_contains_expected_strategies():
    ids = list_strategy_ids()
    assert ids == tuple(sorted({
        "natural_language",
        "json_schema",
        "pseudocode",
        "symbolic_acl",
        "emergent_dsl",
    }))
    assert set(ids) == set(STRATEGY_REGISTRY.keys())


@pytest.mark.parametrize("strategy_id", list_strategy_ids())
def test_strategy_round_trip(strategy_id: str) -> None:
    definition = get_strategy_definition(strategy_id)
    strategy = build_strategy(definition)

    assert strategy.id == definition.id
    assert strategy.output_mode == definition.output_mode
    assert strategy.agent_defaults.max_new_tokens == definition.agent_profile.max_new_tokens

    decorated = strategy.decorate_prompts("Solve the task", {"agent": "tester"})
    if strategy.json_only:
        assert "JSON" in decorated.upper()
    else:
        assert "JSON" not in decorated[:20].upper()

    context = {}
    strategy.apply_pre_round_hooks(context)
    strategy.apply_controller_behaviors(context)
    assert context.get("meta", {}).get("strategy_id") == strategy_id


def test_build_strategy_overrides_decoding():
    strategy = build_strategy("json_schema", overrides={"decoding": {"max_new_tokens": 128}})
    assert strategy.decoding["max_new_tokens"] == 128
    assert strategy.decoding["temperature"] == 0.0


def test_metadata_exposed_through_registry():
    strat = REGISTRY["pseudocode"].instantiate()
    assert strat.metadata["requires_pseudocode"] is True
