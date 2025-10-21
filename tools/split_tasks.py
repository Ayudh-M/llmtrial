import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', default='tasks.jsonl')
    ap.add_argument('--parts', type=int, default=4)
    ap.add_argument('--out_prefix', default='tasks_shard')
    args = ap.parse_args()

    tasks = [json.loads(l) for l in Path(args.tasks).read_text(encoding='utf-8').splitlines() if l.strip()]
    n = max(1, args.parts)
    shards = [[] for _ in range(n)]
    for i, t in enumerate(tasks):
        shards[i % n].append(t)

    for i, shard in enumerate(shards):
        outp = f"{args.out_prefix}{i}.jsonl"
        with open(outp, 'w', encoding='utf-8') as w:
            for t in shard:
                w.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"Shard {i}: {len(shard)} -> {outp}")
    print(f"Total tasks: {len(tasks)} split into {n} shards.")

if __name__ == "__main__":
    main()
