from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.template_loader import load_registry


def _parse_reps(spec: str | None) -> List[int]:
    if not spec:
        return [0]
    reps: List[int] = []
    for chunk in spec.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start_str, _, end_str = token.partition("-")
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError as exc:  # pragma: no cover - defensive
                raise SystemExit(f"Invalid rep range: {token}") from exc
            if end < start:
                raise SystemExit(f"Invalid rep range: {token}")
            reps.extend(range(start, end + 1))
        else:
            try:
                reps.append(int(token))
            except ValueError as exc:  # pragma: no cover - defensive
                raise SystemExit(f"Invalid rep value: {token}") from exc
    if not reps:
        reps.append(0)
    return sorted(set(reps))


def _scenario_records(reps: Iterable[int]) -> List[Dict[str, object]]:
    registry = load_registry()["scenarios"]
    records: List[Dict[str, object]] = []
    for key in sorted(registry):
        parts = key.split(":")
        if len(parts) < 3:
            continue
        dataset = parts[0]
        language = parts[1]
        pair = ":".join(parts[2:])
        if ":rep=" in pair:
            pair, _, _ = pair.partition(":rep=")
        for rep in reps:
            records.append(
                {
                    "dataset": dataset,
                    "language": language,
                    "pair": pair,
                    "rep": rep,
                }
            )
    return records


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False)
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build flat task list for minimal duet runner")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--reps", help="Comma-separated reps (supports ranges like 0-2)")
    parser.add_argument(
        "--shard-size",
        type=int,
        default=0,
        help="Optional shard size; if provided, emit numbered shard files next to --out",
    )
    args = parser.parse_args()

    reps = _parse_reps(args.reps)
    rows = _scenario_records(reps)

    out_path = Path(args.out)
    _write_jsonl(out_path, rows)

    shard_size = int(args.shard_size or 0)
    if shard_size > 0:
        total = len(rows)
        base = out_path.with_suffix("")
        for idx in range(0, total, shard_size):
            shard_rows = rows[idx : idx + shard_size]
            shard_path = base.parent / f"{base.name}_shard{idx // shard_size}.jsonl"
            _write_jsonl(shard_path, shard_rows)

    print(f"Wrote {len(rows)} tasks to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
