"""Utilities for recording structured run data.

The new analytics surface keeps both JSONL and CSV snapshots for every
controller execution so downstream analysis can pivot across strategies,
rolesets, and model selections without re-running experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping
import csv
import json


@dataclass
class RunMetadata:
    """Descriptor for a controller execution."""

    scenario_id: str
    roleset: str
    strategy_id: str
    model_a: str
    model_b: str
    extra: Dict[str, Any] = field(default_factory=dict)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: str | Path, obj: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _flatten_intent_counts(intents: Mapping[str, Mapping[str, int]]) -> Dict[str, int]:
    flattened: Dict[str, int] = {}
    for actor, counts in intents.items():
        for intent, value in counts.items():
            key = f"intent_{actor}_{intent}".lower()
            flattened[key] = int(value)
    return flattened


def build_run_record(result: Mapping[str, Any], meta: RunMetadata) -> Dict[str, Any]:
    """Merge controller output with metadata for logging.

    The record favours simple scalars so that the CSV stays tidy while the
    companion JSONL retains the richer nested payloads for debugging.
    """

    transcript = result.get("transcript") or []
    analytics = result.get("analytics") or {}
    intents = analytics.get("intent_counts") or {}

    record: Dict[str, Any] = {
        "timestamp": now_iso(),
        "scenario_id": meta.scenario_id,
        "roleset": meta.roleset,
        "strategy_id": meta.strategy_id,
        "model_a": meta.model_a,
        "model_b": meta.model_b,
        "status": result.get("status", "UNKNOWN"),
        "rounds": result.get("rounds") or len(transcript),
        "canonical_text": result.get("canonical_text") or "",
        "sha256": result.get("sha256") or "",
        "transcript_turns": len(transcript),
    }

    # Include consensus helper flags and optional final actor metadata when present.
    final_message = result.get("final_message") or {}
    if isinstance(final_message, Mapping):
        record["final_actor"] = final_message.get("actor")
        dsl = final_message.get("dsl") or {}
        if isinstance(dsl, Mapping):
            record["final_canonical"] = dsl.get("canonical_text") or ""

    record.update(_flatten_intent_counts(intents))
    record.update(meta.extra)
    return record


def append_csv(path: str | Path, fieldnames: Iterable[str], row: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    file_exists = p.exists()
    with p.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def record_run(
    result: Mapping[str, Any],
    meta: RunMetadata,
    *,
    csv_path: str | Path,
    jsonl_path: str | Path,
) -> Dict[str, Any]:
    """Persist run analytics to both CSV and JSONL formats.

    Returns the flattened record for callers that wish to perform additional
    in-memory analysis without re-flattening the payload.
    """

    record = build_run_record(result, meta)
    append_jsonl(jsonl_path, {"record": record, "raw_result": result})

    # Deterministic ordering keeps CSV columns stable across runs.
    fieldnames = sorted(record.keys())
    append_csv(csv_path, fieldnames, record)
    return record


__all__ = [
    "RunMetadata",
    "append_jsonl",
    "append_csv",
    "build_run_record",
    "now_iso",
    "record_run",
]
