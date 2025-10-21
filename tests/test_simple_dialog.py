# -*- coding: utf-8 -*-
from __future__ import annotations

from src.simple_agents import GenConfig
from src.simple_dialog import run_dialog


class DummyHF:
    def __init__(self, outputs):
        self.outputs = outputs
        self.i = 0

    def respond(self, system_prompt, incoming, cfg):
        out = self.outputs[self.i % len(self.outputs)]
        self.i += 1
        return out, 10, 5, "eos_or_sample"


def test_fixed_turn_pass_through(monkeypatch, tmp_path):
    import src.simple_dialog as sd

    agent_a = DummyHF(outputs=["plan A1", "plan A2", "plan A3"])
    agent_b = DummyHF(outputs=["solve B1", "solve B2", "solve B3"])
    monkeypatch.setattr(sd, "SimpleHF", lambda model_id: agent_a if "modelA" in model_id else agent_b)

    result = run_dialog(
        scenario_text="TASK: add 2+2",
        strategy_id="NL",
        roleset_id="Planner-Solver",
        turns=4,
        model_a="modelA",
        model_b="modelB",
        gen_cfg=GenConfig(max_new_tokens=32, do_sample=False),
        seed=123,
        outdir=str(tmp_path),
    )

    transcript = result["transcript"]
    cfg = result["config"]

    assert len(transcript) == 4
    assert transcript[0]["actor"] == "A"
    assert transcript[1]["actor"] == "B"
    assert transcript[1]["text_in"] == transcript[0]["text_out"]
    assert cfg["turns"] == 4
