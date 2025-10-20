from types import SimpleNamespace

from src.main import (
    _actor_display_name,
    _condense_text,
    _format_envelope,
    _print_run_summary,
    _print_transcript,
)


def test_actor_display_name_prefers_agent_names(capsys):
    agent_pair = (SimpleNamespace(name="Planner"), SimpleNamespace(name="Solver"))
    assert _actor_display_name("a", agent_pair) == "Planner"
    assert _actor_display_name("beta", agent_pair) == "Solver"
    assert _actor_display_name("Judge", agent_pair) == "Judge"
    assert _actor_display_name(None, agent_pair) == "Unknown"


def test_condense_text_truncates_long_messages():
    text = "Hello\nworld   this is\ta test"
    assert _condense_text(text) == "Hello world this is a test"
    long_text = "A" * 150
    condensed = _condense_text(long_text, width=20)
    assert condensed.endswith("…")
    assert len(condensed) == 20


def test_format_envelope_and_print_transcript(capsys):
    agent_pair = (SimpleNamespace(name="Planner"), SimpleNamespace(name="Solver"))
    transcript = [
        {
            "r": 1,
            "actor": "a",
            "envelope": {"tag": "[CONTACT]", "status": "PROPOSED"},
            "raw": "Line one\nLine two",
        },
        {
            "r": 2,
            "actor": "b",
            "envelope": {"tag": "[SOLVED]", "status": "SOLVED"},
            "raw": "Final answer",
        },
        {
            "r": 3,
            "actor": "judge",
            "envelope": None,
            "raw": None,
        },
    ]

    assert _format_envelope(transcript[0]["envelope"]) == "[CONTACT]/PROPOSED"
    assert _format_envelope({"status": "WORKING"}) == "WORKING"
    assert _format_envelope(None) == ""

    _print_transcript(transcript, agent_pair)
    out = capsys.readouterr().out
    assert "[round 1] Planner [[CONTACT]/PROPOSED]" in out
    assert "Line one Line two" in out
    assert "[round 2] Solver [[SOLVED]/SOLVED] Final answer" in out
    assert "[round 3] judge" in out


def test_print_run_summary_includes_final_message_details(capsys):
    agent_pair = (SimpleNamespace(name="Planner"), SimpleNamespace(name="Solver"))
    record = {
        "rounds": 5,
        "transcript_turns": 10,
        "final_actor": "b",
        "final_canonical": "ANSWER: 60 km/h",
    }
    result = {
        "transcript": [{}] * 10,
        "final_message": {
            "actor": "b",
            "envelope": {"tag": "[SOLVED]", "status": "SOLVED"},
            "raw": "ACCEPT: ANSWER => DONE",
        },
    }

    _print_run_summary(record, result, agent_pair)
    out = capsys.readouterr().out
    assert "rounds=5" in out
    assert "turns=10" in out
    assert "final_actor=Solver" in out
    assert "final_canonical=ANSWER: 60 km/h" in out
    assert "final_message=Solver [[SOLVED]/SOLVED] ACCEPT:" in out


def test_print_run_summary_handles_missing_fields(capsys):
    agent_pair = (SimpleNamespace(name="Planner"), SimpleNamespace(name="Solver"))
    record = {}
    result = {"transcript": []}

    _print_run_summary(record, result, agent_pair)
    out = capsys.readouterr().out
    assert "rounds=0" in out
    assert "turns=0" in out
    assert "final_actor=<none>" in out
    assert "final_canonical=<none>" in out
    assert "final_message=<none>" in out
