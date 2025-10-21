# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml

from src.presets import STRATEGIES
from src.simple_dialog import run_dialog


def _git_rev() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def load_tasks(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("tasks", [])


def detect_first_hit(transcript: List[Dict[str, Any]], regex: Optional[str]) -> Optional[int]:
    if not regex:
        return None
    pat = re.compile(regex, flags=re.I)
    for turn in transcript:
        text_out = (turn.get("text_out") or "").replace(",", "")
        if pat.search(text_out):
            return turn.get("r")
    return None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="tasks/tasks.yaml")
    ap.add_argument("--strategies", default="ALL", help="Comma list or ALL")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--model-a", required=True)
    ap.add_argument("--model-b", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat each (task,strategy) N times with different seeds",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed; each repeat adds +k",
    )
    ap.add_argument("--outdir", default="logs/matrix")
    args = ap.parse_args()

    tasks = load_tasks(args.tasks)
    if args.strategies == "ALL":
        strategies = sorted(STRATEGIES)
    else:
        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root_out = os.path.join(args.outdir, f"matrix_{stamp}")
    ensure_dir(root_out)

    master_csv = os.path.join(root_out, "matrix_results.csv")
    fieldnames = [
        "timestamp",
        "git_rev",
        "task_id",
        "strategy",
        "roleset",
        "model_a",
        "model_b",
        "turns",
        "repeat_idx",
        "seed",
        "elapsed_sec",
        "total_prompt_tokens",
        "total_output_tokens",
        "tokens_per_sec",
        "first_hit_turn",
        "last_stop_reason",
        "out_jsonl",
        "out_csv",
    ]
    with open(master_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

    total_runs = 0
    for task in tasks:
        task_id = task["id"]
        scenario = task["scenario"]
        roleset = task["roleset"]
        ans_regex = task.get("answer_regex")

        for strat in strategies:
            for rep in range(args.repeats):
                seed = args.seed + rep
                run_id = f"{task_id}_{strat}_rep{rep}"
                sub_out = os.path.join(root_out, run_id)
                ensure_dir(sub_out)

                loop_start = time.time()
                out = run_dialog(
                    scenario=scenario,
                    strategy=strat,
                    roleset=roleset,
                    turns=args.turns,
                    model_a=args.model_a,
                    model_b=args.model_b,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=seed,
                    outdir=sub_out,
                )
                elapsed = out.get("elapsed_sec") or (time.time() - loop_start)
                cfg = out["config"]
                transcript = out["transcript"]

                total_prompt = int(cfg.get("total_prompt_tokens", 0))
                total_output = int(cfg.get("total_output_tokens", 0))
                tps = (total_prompt + total_output) / max(elapsed, 1e-6)
                last_stop = transcript[-1].get("stop_reason") if transcript else None
                first_hit = detect_first_hit(transcript, ans_regex)

                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "git_rev": _git_rev(),
                    "task_id": task_id,
                    "strategy": strat,
                    "roleset": roleset,
                    "model_a": args.model_a,
                    "model_b": args.model_b,
                    "turns": args.turns,
                    "repeat_idx": rep,
                    "seed": seed,
                    "elapsed_sec": round(elapsed, 3),
                    "total_prompt_tokens": total_prompt,
                    "total_output_tokens": total_output,
                    "tokens_per_sec": round(tps, 2),
                    "first_hit_turn": first_hit,
                    "last_stop_reason": last_stop,
                    "out_jsonl": out["out_jsonl"],
                    "out_csv": out["out_csv"],
                }

                with open(master_csv, "a", newline="", encoding="utf-8") as handle:
                    csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)

                total_runs += 1
                print(
                    f"[{total_runs}] {task_id} × {strat} rep{rep} -> "
                    f"{row['out_jsonl']} ({elapsed:.1f}s)"
                )

    print("WROTE", master_csv)


if __name__ == "__main__":
    main()
