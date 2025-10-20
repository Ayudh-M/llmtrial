"""Display a summary of the most recent run artifact."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parents[1] / "runs"


def main() -> None:
    files = sorted(glob.glob(str(RUN_DIR / "*.json")), key=os.path.getmtime)
    if not files:
        raise SystemExit("No runs/*.json artifacts found. Run a scenario first.")

    latest = Path(files[-1])
    print(f"LATEST: {latest}")
    data = json.loads(latest.read_text(encoding="utf-8"))
    status = data.get("status")
    rounds = data.get("rounds")
    canonical = data.get("canonical_text")
    print(f"status={status} rounds={rounds} canonical_text={canonical}")

    transcript = data.get("transcript") or []
    if not transcript:
        return

    print("-- transcript summary --")
    for idx, row in enumerate(transcript, 1):
        actor = row.get("actor") or row.get("actor_key")
        if "envelope" in row:
            envelope = row["envelope"]
            final_solution = (envelope.get("final_solution") or {}).get("canonical_text")
            verdict = (envelope.get("content") or {}).get("verdict")
            status = envelope.get("status")
            tag = envelope.get("tag")
            print(f"round {idx:02d} {actor}: status={status} tag={tag} canonical={final_solution} verdict={verdict}")
        elif "text" in row:
            snippet = str(row["text"]).splitlines()[0]
            print(f"round {idx:02d} {actor}: text={snippet}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
