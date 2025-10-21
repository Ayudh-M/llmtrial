#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-logs}"
MODE="${2:---archive}"     # --archive (default) or --delete
EXCLUDE="${3:-}"           # optional: absolute or glob path to exclude, e.g. logs/matrix_fullrerun_20251021-193000*

TS="$(date +%Y%m%d-%H%M%S)"
if [[ "$MODE" == "--archive" ]]; then
  mkdir -p "$ROOT/_archive/$TS"
fi

shopt -s nullglob
moved=0
for d in "$ROOT"/matrix_*; do
  [[ "$d" == "$ROOT/_archive"* ]] && continue
  [[ -n "$EXCLUDE" && "$d" == $EXCLUDE ]] && continue

  if [[ -d "$d" ]]; then
    if [[ "$MODE" == "--delete" ]]; then
      rm -rf "$d"
      echo "Deleted $d"
    else
      mv "$d" "$ROOT/_archive/$TS/"
      echo "Archived $d -> $ROOT/_archive/$TS/"
    fi
    moved=$((moved+1))
  fi
done
echo "Rotate complete ($moved dirs)."
