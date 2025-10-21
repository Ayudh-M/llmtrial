import argparse
import json
import re
from pathlib import Path

DIR_RE = re.compile(r'^(?P<dataset>.+)_(?P<language>[A-Z_]+)_rep(?P<rep>\d+)$')
FILE_RE = re.compile(r'^runs_fixed_(?P<language>[A-Z_]+)_(?P<pair>.+?)_\d{8}-\d{6}\.jsonl$')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', required=True, help='e.g., logs/matrix_20251021-180517')
    ap.add_argument('--out', default='tasks.jsonl')
    args = ap.parse_args()

    root = Path(args.logs)
    seen = set()
    out = []

    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        mdir = DIR_RE.match(d.name)
        if not mdir:
            continue
        dataset = mdir.group('dataset')
        language = mdir.group('language')
        rep = int(mdir.group('rep'))

        for jf in sorted(d.glob('runs_fixed_*.jsonl')):
            mf = FILE_RE.match(jf.name)
            if not mf:
                continue
            pair = mf.group('pair')
            key = (dataset, language, pair, rep)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "dataset": dataset,
                "language": language,
                "pair": pair,
                "rep": rep
            })

    with open(args.out, 'w', encoding='utf-8') as w:
        for t in out:
            w.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out)} tasks -> {args.out}")

if __name__ == "__main__":
    main()
