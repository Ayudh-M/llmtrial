from src.template_loader import get_scenario


def test_get_scenario_prefers_exact_over_rep(monkeypatch):
    fake_registry = {
        "scenarios": {
            "A:B:C": {"marker": "base"},
            "A:B:C:rep=1": {"marker": "exact"},
        }
    }
    monkeypatch.setattr("src.template_loader.load_registry", lambda: fake_registry)

    scenario = get_scenario("A:B:C:rep=1")
    assert scenario["marker"] == "exact"

    fallback = get_scenario("A:B:C:rep=999")
    assert fallback["marker"] == "base"
