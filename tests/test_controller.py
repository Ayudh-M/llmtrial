from src.controller import run_controller
from src.agents_mock import MockAgent, ConciseTextAgent
from src.strategies import build_strategy

def test_mock_consensus():
    a = MockAgent("A", "TRUE")
    b = MockAgent("B", "TRUE")
    out = run_controller("Return TRUE", a, b, max_rounds=4, kind=None)
    assert out["status"] == "CONSENSUS"
    assert out["canonical_text"] == "TRUE"


def test_concise_text_strategy_transcript():
    strat = build_strategy(
        {
            "id": "S_concise",
            "name": "concise_text",
            "json_only": False,
            "validator": "concise_text",
            "validator_params": {"max_sentences": 1, "max_tokens": 6},
            "envelope_required": False,
            "decoding": {"do_sample": False, "temperature": 0.3, "max_new_tokens": 32},
            "prompt_snippet": "Keep answers compact.",
        }
    )
    a = ConciseTextAgent("A", ["  Ready."], "42")
    b = ConciseTextAgent("B", [" Standing by.  "], "42")
    out = run_controller("Return 42", a, b, max_rounds=3, kind=None, strategy=strat)

    assert out["status"] == "CONSENSUS"
    assert out["canonical_text"] == "42"
    texts = [entry["envelope"]["text"] for entry in out["transcript"]]
    for text in texts:
        assert text == text.strip()
        assert len(text.split()) <= 6
