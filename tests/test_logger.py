import json
from pathlib import Path

import pytest

from src.logger import RunMetadata, build_run_record, record_run


@pytest.fixture()
def sample_result():
    return {
        "status": "CONSENSUS",
        "rounds": 3,
        "canonical_text": "TRUE",
        "sha256": "abc",
        "transcript": [{}, {}, {}],
        "final_message": {"actor": "b", "dsl": {"canonical_text": "TRUE"}},
        "analytics": {"intent_counts": {"a": {"SOLVED": 1}, "b": {"SOLVED": 2}}},
    }


def test_build_run_record_flattens_intents(sample_result):
    meta = RunMetadata(
        scenario_id="demo",
        roleset="rolesets/demo.json",
        strategy_id="S1",
        model_a="model/a",
        model_b="model/b",
    )
    record = build_run_record(sample_result, meta)
    assert record["intent_a_solved"] == 1
    assert record["intent_b_solved"] == 2
    assert record["final_actor"] == "b"
    assert record["final_canonical"] == "TRUE"


def test_record_run_persists_csv_and_jsonl(tmp_path: Path, sample_result):
    meta = RunMetadata(
        scenario_id="demo",
        roleset="rolesets/demo.json",
        strategy_id="S2",
        model_a="model/a",
        model_b="model/b",
        extra={"batch_id": "exp-1"},
    )
    csv_path = tmp_path / "runs" / "summary.csv"
    jsonl_path = tmp_path / "runs" / "summary.jsonl"

    record = record_run(sample_result, meta, csv_path=csv_path, jsonl_path=jsonl_path)

    assert csv_path.exists()
    with csv_path.open("r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    assert len(lines) == 2  # header + row
    assert "batch_id" in lines[0]

    assert jsonl_path.exists()
    with jsonl_path.open("r", encoding="utf-8") as f:
        payload = json.loads(f.readline())
    assert payload["record"]["batch_id"] == "exp-1"
    assert payload["raw_result"]["status"] == "CONSENSUS"
    assert record["batch_id"] == "exp-1"
