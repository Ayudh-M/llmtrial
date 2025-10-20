from __future__ import annotations

import json
from pathlib import Path

from src import main as entry


def test_main_mock_creates_consensus_artifact(tmp_path, capsys):
    output_dir = tmp_path / "runs"
    entry.main([
        "--scenario",
        "mistral_math_smoke",
        "--mock",
        "--output-dir",
        str(output_dir),
    ])

    captured = capsys.readouterr().out
    assert "CONSENSUS" in captured

    artifacts = list(Path(output_dir).glob("*.json"))
    assert len(artifacts) == 1

    payload = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert payload["status"] == "CONSENSUS"
    assert payload["canonical_text"] == "42"
    assert payload["final_messages"]["a"]["envelope"]["final_solution"]["canonical_text"] == "42"
