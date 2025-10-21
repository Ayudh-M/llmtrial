import argparse
import json
import os
import shlex
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RUN_TEMPLATE = os.environ.get(
    "LLMTRIAL_RUN_TEMPLATE",
    "python -m llmtrial.run_one --dataset {dataset} --language {language} --pair {pair} --rep {rep} --logdir {logdir}"
)

def ensure_fresh_logdir(path: Path, fresh: bool):
    if fresh and path.exists():
        if not str(path).startswith("logs/") or "matrix_" not in path.name:
            raise SystemExit(f"Refusing to delete suspicious logdir: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', required=True)
    ap.add_argument('--logdir', required=True)
    ap.add_argument('--max_workers', type=int, default=4)
    ap.add_argument('--fresh', action='store_true', help='If set, remove existing logdir first (safe-guarded).')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    logdir = Path(args.logdir)
    ensure_fresh_logdir(logdir, args.fresh)

    tasks = []
    with open(args.tasks, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    print(f"Loaded {len(tasks)} tasks from {args.tasks}")

    cmds = [RUN_TEMPLATE.format(logdir=str(logdir), **t) for t in tasks]
    for c in cmds:
        print("CMD:", c)

    if args.dry_run or not cmds:
        if args.dry_run:
            print("Dry run; exiting.")
        return

    def run(cmd):
        return cmd, subprocess.run(shlex.split(cmd), capture_output=True, text=True)

    ok_count = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(run, c) for c in cmds]
        for fut in as_completed(futures):
            cmd, res = fut.result()
            ok = res.returncode == 0
            ok_count += int(ok)
            print(f"[{'OK' if ok else 'ERR'}] {cmd}")
            if not ok:
                print("STDOUT:\n", res.stdout)
                print("STDERR:\n", res.stderr)
    print(f"Done. {ok_count}/{len(cmds)} succeeded.")

if __name__ == "__main__":
    main()
