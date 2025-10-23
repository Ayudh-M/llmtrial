from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_TEMPLATE = os.environ.get(
    "LLMTRIAL_RUN_TEMPLATE",
    "python -m src.minimal_duet --dataset {dataset} --language {language} --pair {pair} --rep {rep} --logdir {logdir}",
)


def _read_tasks(path: Path) -> List[dict]:
    tasks: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            tasks.append(json.loads(text))
    return tasks


def _format_commands(tasks: Iterable[dict], logdir: Path) -> List[str]:
    commands: List[str] = []
    for row in tasks:
        try:
            command = RUN_TEMPLATE.format(logdir=str(logdir), **row)
        except KeyError as exc:  # pragma: no cover - defensive
            missing = exc.args[0]
            raise SystemExit(f"Task row missing field: {missing}") from exc
        commands.append(command)
    return commands


def _ensure_logdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _stream_process(command: str, log_file: Path) -> int:
    process = subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None  # for type checkers

    with log_file.open("a", encoding="utf-8") as handle:
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        process.stdout.close()
        process.wait()
        status = process.returncode
        status_tag = "OK" if status == 0 else "ERR"
        summary = f"[{status_tag}] {command}\n"
        print(summary, end="")
        handle.write(summary)
    return process.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal duet tasks sequentially")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--max_workers", type=int, default=1, help="Reserved for compatibility; runs sequentially")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    logdir = Path(args.logdir)
    _ensure_logdir(logdir)

    tasks = _read_tasks(tasks_path)
    print(f"Loaded {len(tasks)} tasks from {tasks_path}")

    commands = _format_commands(tasks, logdir)
    for cmd in commands:
        print("CMD:", cmd)

    if args.dry_run or not commands:
        if args.dry_run:
            print("Dry run; exiting.")
        return

    runner_log = logdir / "runner.log"
    for cmd in commands:
        ret = _stream_process(cmd, runner_log)
        if ret != 0:
            print(f"Command failed with exit code {ret}")


if __name__ == "__main__":  # pragma: no cover
    main()
