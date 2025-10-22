from src.strategies import list_strategy_ids


_DEF_STYLE_IDS = {"DSL", "JSON_SCHEMA", "PSEUDOCODE", "KQMLISH", "EMERGENT_TOY", "NL"}


def test_style_strategies_registered():
    strategy_ids = set(list_strategy_ids())
    missing = _DEF_STYLE_IDS - strategy_ids
    assert not missing, f"Missing strategies: {sorted(missing)}"
