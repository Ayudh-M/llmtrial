#!/bin/bash
#SBATCH --job-name=llm-matrix
#SBATCH --partition=${PART:-thin}          # set via sbatch: --export=ALL,PART=...
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS:-8}
#SBATCH --mem=${MEM:-64G}
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail
mkdir -p slurm_logs

: "${TASKS:?Need TASKS jsonl path (export TASKS=...)}"
: "${LOGDIR:?Need LOGDIR (export LOGDIR=...)}"
WORKERS="${WORKERS:-8}"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate llmtrial 2>/dev/null || true

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

echo "Running shard: $TASKS"
echo "Logdir:       $LOGDIR"
echo "Workers:      $WORKERS"

python scripts/run_tasks.py --tasks "$TASKS" --logdir "$LOGDIR" --max_workers "$WORKERS"
