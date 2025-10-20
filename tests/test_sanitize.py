import pytest

from src.sanitize import NON_SOLVED_FINALS, repair_envelope


def test_repair_envelope_preserves_plan_signal_without_marking_solved() -> None:
    raw = {
        "tag": "[plan]",
        "status": "proposed",
        "final_solution": {"canonical_text": "PLAN_READY"},
    }

    fixed = repair_envelope(raw)

    assert fixed["tag"] == "[PLAN]"
    assert fixed["status"] == "PROPOSED"
    assert fixed["final_solution"]["canonical_text"] == "PLAN_READY"


@pytest.mark.parametrize("final_text", sorted(NON_SOLVED_FINALS))
def test_repair_envelope_downgrades_solved_status_for_plan_like_finals(final_text: str) -> None:
    raw = {
        "tag": "[solved]",
        "status": "SOLVED",
        "final_solution": {"canonical_text": final_text},
    }

    fixed = repair_envelope(raw)

    assert fixed["tag"] == "[CONTACT]"
    assert fixed["status"] == "PROPOSED"
    assert fixed["final_solution"]["canonical_text"] == final_text


def test_repair_envelope_keeps_non_solved_status_when_contact() -> None:
    raw = {
        "tag": "[contact]",
        "status": "revised",
        "final_solution": {"canonical_text": "PLAN_NEEDS_WORK"},
    }

    fixed = repair_envelope(raw)

    assert fixed["tag"] == "[CONTACT]"
    assert fixed["status"] == "REVISED"
    assert fixed["final_solution"]["canonical_text"] == "PLAN_NEEDS_WORK"


def test_repair_envelope_forces_solved_status_when_tagged_solved() -> None:
    raw = {
        "tag": "[SOLVED]",
        "status": "proposed",
        "final_solution": {"canonical_text": "ANSWER: 42"},
    }

    fixed = repair_envelope(raw)

    assert fixed["tag"] == "[SOLVED]"
    assert fixed["status"] == "SOLVED"
    assert fixed["final_solution"]["canonical_text"] == "ANSWER: 42"


def test_repair_envelope_defaults_on_invalid_tag_and_missing_final_solution() -> None:
    raw = {
        "tag": "not-a-tag",
        "status": "solved",
        "final_solution": {"canonical_text": " 42 "},
    }

    fixed = repair_envelope(raw)

    assert fixed["tag"] == "[SOLVED]"
    assert fixed["status"] == "SOLVED"
    assert fixed["final_solution"]["canonical_text"] == "42"


def test_repair_envelope_preserves_other_valid_tags() -> None:
    raw = {
        "tag": "[test]",
        "status": "ready_to_solve",
        "final_solution": {},
    }

    fixed = repair_envelope(raw)

    assert fixed["tag"] == "[TEST]"
    assert fixed["status"] == "READY_TO_SOLVE"
    assert "final_solution" not in fixed


def test_repair_envelope_keeps_stage_solved_status_for_code_updates() -> None:
    raw = {
        "tag": "[code]",
        "status": "solved",
        "final_solution": {"canonical_text": "READY"},
    }

    fixed = repair_envelope(raw)

    assert fixed["tag"] == "[CODE]"
    assert fixed["status"] == "SOLVED"
    assert fixed["final_solution"]["canonical_text"] == "READY"


def test_repair_envelope_keeps_stage_solved_status_for_critique_approved() -> None:
    raw = {
        "tag": "[critique]",
        "status": "SOLVED",
        "final_solution": {"canonical_text": "APPROVED"},
    }

    fixed = repair_envelope(raw)

    assert fixed["tag"] == "[CRITIQUE]"
    assert fixed["status"] == "SOLVED"
    assert fixed["final_solution"]["canonical_text"] == "APPROVED"


def test_repair_envelope_strips_extra_final_solution_fields() -> None:
    raw = {
        "tag": "[code]",
        "status": "SOLVED",
        "final_solution": {
            "canonical_text": "READY",
            "sha256": "should-go",
            "notes": "unused",
        },
    }

    fixed = repair_envelope(raw)

    assert fixed["final_solution"] == {"canonical_text": "READY"}


def test_repair_envelope_trims_canonical_text() -> None:
    raw = {
        "tag": "[solved]",
        "status": "solved",
        "final_solution": {"canonical_text": "  ANSWER: 7\n"},
    }

    fixed = repair_envelope(raw)

    assert fixed["final_solution"]["canonical_text"] == "ANSWER: 7"
