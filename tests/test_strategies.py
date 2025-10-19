from __future__ import annotations

import pytest

from src.strategies import (
    STRATEGY_REGISTRY,
    build_strategy,
    get_strategy_definition,
    list_strategy_ids,
)


def test_registry_ids_sorted():
    ids = list_strategy_ids()
    assert ids == tuple(sorted(ids))
    assert set(ids) == set(STRATEGY_REGISTRY.keys())


@pytest.mark.parametrize("strategy_id", list_strategy_ids())
def test_strategy_behaviours(strategy_id: str) -> None:
    definition = get_strategy_definition(strategy_id)
    strategy = build_strategy(definition)

    assert strategy.id == definition.id
    assert strategy.max_rounds == definition.max_rounds

    # Prompt decoration appends a JSON hint.
    prompt = "Solve task"
    decorated = strategy.decorate_prompts(prompt, {})
    assert decorated.endswith("JSON object.")

    # Envelope validation succeeds with required keys and fails otherwise.
    ok, errors = strategy.validate_envelope({"status": "WORKING", "tag": "[CONTACT]"})
    assert ok
    assert errors == []

    ok, errors = strategy.validate_envelope({})
    assert not ok
    assert errors

    # Controller hooks add metadata to the context.
    ctx = {}
    strategy.apply_pre_round_hooks(ctx)
    strategy.apply_controller_behaviors(ctx)
    assert ctx["meta"]["strategy_id"] == strategy_id
    assert ctx["meta"]["json_only"] is True

    # Agent defaults return a copy each time.
    profile = strategy.agent_defaults
    profile2 = strategy.agent_defaults
    assert profile is not profile2
    assert profile.max_new_tokens == definition.agent_profile.max_new_tokens


def test_build_strategy_overrides() -> None:
    strategy = build_strategy({
        "id": "S1",
        "max_rounds": 12,
        "decoding": {"max_new_tokens": 64},
    })
    assert strategy.max_rounds == 12
    assert strategy.decoding["max_new_tokens"] == 64
    # Base decoding keys are preserved.
    assert strategy.decoding["temperature"] == 0.0
