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

    duration = result.get("duration_s")

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

    if isinstance(duration, (int, float)):
        record["duration_s"] = float(duration)

    # Include consensus helper flags and optional final actor metadata when present.
    final_message = result.get("final_message") or {}
    if isinstance(final_message, Mapping):
        record["final_actor"] = final_message.get("actor")
        canonical = final_message.get("canonical_text")
        if not canonical:
            dsl = final_message.get("dsl") or {}
            if isinstance(dsl, Mapping):
                canonical = dsl.get("canonical_text")
        if canonical:
            record["final_canonical"] = canonical

    control_stats = (result.get("analytics") or {}).get("control") or {}
    if isinstance(control_stats, Mapping):
        record["trailer_missing_ct"] = int(control_stats.get("trailer_missing_ct", 0) or 0)
        record["invalid_trailer_ct"] = int(control_stats.get("invalid_trailer_ct", 0) or 0)
        record["retry_count"] = int(control_stats.get("retry_count", 0) or 0)
        record["retries_total"] = record["retry_count"]
        first_error = control_stats.get("first_error")
        if first_error:
            record["first_error"] = first_error
        stopped = control_stats.get("stopped_on_ctrl_ct")
        if isinstance(stopped, (int, float)):
            record["stopped_on_ctrl_ct"] = int(stopped)
        handshake_errors = control_stats.get("handshake_error_ct")
        if isinstance(handshake_errors, (int, float)):
            record["handshake_error_ct"] = int(handshake_errors)
        avg_body = control_stats.get("avg_body_len")
        if isinstance(avg_body, (int, float)):
            record["avg_body_len"] = float(avg_body)
        avg_trailer = control_stats.get("avg_trailer_len")
        if isinstance(avg_trailer, (int, float)):
            record["avg_trailer_len"] = float(avg_trailer)
        avg_reserved = control_stats.get("avg_tokens_reserved")
        if isinstance(avg_reserved, (int, float)):
            record["avg_tokens_reserved"] = float(avg_reserved)
        else:
            record["avg_tokens_reserved"] = 0.0
        first_valid = control_stats.get("first_valid_round")
        if isinstance(first_valid, int):
            record["first_valid_round"] = first_valid
        first_proposal = control_stats.get("first_proposal_round")
        if isinstance(first_proposal, int):
            record["first_proposal_round"] = first_proposal
        solved_round = control_stats.get("solved_round")
        record["solved_round"] = int(solved_round) if isinstance(solved_round, int) else 0
        proposer = control_stats.get("proposer")
        if proposer:
            record["proposer"] = proposer
        acceptor = control_stats.get("acceptor")
        if acceptor:
            record["acceptor"] = acceptor
        final_canonical = control_stats.get("final_canonical")
        if final_canonical:
            record["final_canonical"] = final_canonical
        for key, value in control_stats.items():
            if isinstance(key, str) and key.startswith("stop_reason_") and isinstance(value, (int, float)):
                record[key] = int(value)
        record["stopped_on_ctrl"] = int(control_stats.get("stopped_on_ctrl", 0) or 0)
        record["stopped_on_eos"] = int(control_stats.get("stopped_on_eos", 0) or 0)
        stopped_max = control_stats.get("stopped_on_max_new_tokens") or control_stats.get("stopped_on_max_new")
        record["stopped_on_max_new"] = int(stopped_max) if isinstance(stopped_max, (int, float)) else 0
        record["legacy_turns"] = int(control_stats.get("legacy_turns", 0) or 0)
        record["overflow_turns"] = int(control_stats.get("overflow_turns", 0) or 0)
        max_overflow = control_stats.get("max_overflow")
        record["max_overflow"] = float(max_overflow) if isinstance(max_overflow, (int, float)) else 0.0
        needs_reserve = control_stats.get("needs_higher_reserve")
        record["needs_higher_reserve"] = bool(needs_reserve) if isinstance(needs_reserve, bool) else False
        tokens_used_trailer = control_stats.get("tokens_used_trailer_total")
        record["tokens_used_trailer_total"] = (
            float(tokens_used_trailer) if isinstance(tokens_used_trailer, (int, float)) else 0.0
        )
        tokens_used_body = control_stats.get("tokens_used_body_total")
        record["tokens_used_body_total"] = (
            float(tokens_used_body) if isinstance(tokens_used_body, (int, float)) else 0.0
        )
        tokens_used_total = control_stats.get("tokens_used_total")
        record["tokens_used_total"] = (
            float(tokens_used_total) if isinstance(tokens_used_total, (int, float)) else 0.0
        )
        error_counts = control_stats.get("error_counts")
        if isinstance(error_counts, Mapping):
            for key, value in error_counts.items():
                if isinstance(key, str) and isinstance(value, (int, float)):
                    record[f"errors_{key}"] = int(value)

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
