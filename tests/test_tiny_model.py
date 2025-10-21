import os, pytest
from src.model_loader import load_model_and_tokenizer, TINY_REPO
from src.agents_hf import HFChatAgent
from src.strategies import Strategy
from src.controller import run_controller

@pytest.mark.integration
def test_tinystories_runs_greedy():
    try:
        mdl, tok = load_model_and_tokenizer(TINY_REPO, dtype="fp32")
    except Exception as e:
        pytest.skip(f"Tokenizer/model not available: {e}")
    sys = "You are an agent that must output ONLY a JSON object with tag and status. When confident, solve."
    agent = HFChatAgent("A", sys, tok, mdl, Strategy(name="S1", json_only=True, max_rounds=2, decoding={'temperature':0.0,'max_new_tokens':64}))
    env, raw = agent.step("Return ONLY JSON", [])
    assert isinstance(raw, str)
