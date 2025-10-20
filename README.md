# Snellius Clean-Start & Run Guide

This is a copy-paste friendly checklist for teammates to **start clean** on Snellius, clone the repo, set up a fresh environment, and run a job on **H100**. It uses the **2025** software stack.

> If your repo URL is different, change `REPO_URL` below before you run the commands.

---

## 0) One-time: log in

SSH into Snellius from your terminal:

```bash
ssh your_username@snellius.surf.nl
```

---

## 1) Start clean (cancel jobs, remove old checkout, venv, and caches)

```bash
# Cancel any queued/running jobs under your user
squeue -u "$USER" | awk 'NR>1{print $1}' | xargs -r scancel

# Remove previous checkout, venv, and HF caches (user-space only)
rm -rf ~/projects/llmtrial ~/.venvs/consensus ~/.cache/huggingface ~/.cache/hf
```

---

## 2) Clone the repo

```bash
REPO_URL="https://github.com/Ayudh-M/llmtrial.git"   # change if needed
mkdir -p ~/projects && cd ~/projects
git clone "$REPO_URL" llmtrial
cd llmtrial
git remote -v
```

---

## 3) Load modules (2025 stack) and create a fresh venv

```bash
module purge
module load 2025
module load Python/3.12.3-GCCcore-13.3.0

python -m venv ~/.venvs/consensus
source ~/.venvs/consensus/bin/activate

python -m pip install -U pip wheel
pip install -r requirements.txt
```

---

## 4) Quick smoke test (CPU, mock agents)

This does **not** download models. Good to verify your setup.

```bash
python -m src.main --scenario mistral_math_smoke --mock
```

Expected: prints a small **FINAL TEXT** (e.g., `4`).

---

## 5) H100 job script (quoted heredoc; safe to paste)

```bash
cat > run_gpu_h100_only.job <<'SLURM'
#!/bin/bash
#SBATCH -J consensus_h100_only
#SBATCH -A tesr108469
#SBATCH -p gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module purge
module load 2025
module load Python/3.12.3-GCCcore-13.3.0
source "${HOME}/.venvs/consensus/bin/activate"

# Use node-local scratch for Hugging Face cache (faster pulls)
export HF_HOME="${TMPDIR:-/scratch-local/$USER/${SLURM_JOB_ID}}/hf"
mkdir -p "$HF_HOME"

# Ensure local package imports work
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"

# Ensure this is an H100 GPU
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true)"
case "$GPU_NAME" in
  *H100* ) echo "OK: $GPU_NAME";;
  * ) echo "ERROR: Need H100, got: ${GPU_NAME:-unknown}"; exit 2;;
esac

SCENARIO_ID="${1:?usage: sbatch run_gpu_h100_only.job <scenario_id>}"
srun --ntasks=1 python -m src.main --scenario "$SCENARIO_ID"
SLURM
chmod +x run_gpu_h100_only.job
```

> If your SLURM account is not `tesr108469`, change `#SBATCH -A` accordingly.

---

## 6) Submit a GPU run (H100)

```bash
JID=$(sbatch run_gpu_h100_only.job mistral_math_smoke | awk '{print $4}'); echo "JOBID=$JID"
tail -f "logs/consensus_h100_only-$JID.out"
```

---

## 7) Inspect the latest run result

**Option A (if the helper script exists):**

```bash
python scripts/show_latest_run.py
```

**Option B (portable inline snippet):**

```bash
python - <<'PY'
import glob, json, os
files = sorted(glob.glob('runs/*.json'), key=os.path.getmtime)
if not files: raise SystemExit("No runs found.")
p = files[-1]
print("LATEST:", p)
d = json.load(open(p, encoding='utf-8'))
print("status:", d.get("status"), "rounds:", d.get("rounds"))
print("canonical_text:", d.get("canonical_text"))
PY
```

---

## 8) Rerun with a different scenario (examples)

Edit or add scenarios in `prompts/registry.yaml`. Then:

```bash
# Regex example (if defined in registry)
JID=$(sbatch run_gpu_h100_only.job regex_email_basic | awk '{print $4}'); echo "JOBID=$JID"
tail -f "logs/consensus_h100_only-$JID.out"
```

You can override model/dtype in `src.main` if needed (e.g., `--model-a`, `--model-b`, `--dtype`), but by default the **strategy and models** come from the registry.

---

## 9) Common gotchas

* **Do not paste angle brackets** in commands (e.g., `<your-username>`). Use real values or variables as shown.
* Always use the **quoted heredoc** (`<<'SLURM'`) when creating job scripts to prevent accidental interpolation.
* Keep `PYTHONPATH` pointing at `src/` inside the job script.
* Use the **2025** stack modules as above; the 2024 stack is limited and may miss dependencies.

---

## 10) Reset again later (clean slate)

If you want to fully reset and reclone later:

```bash
squeue -u "$USER" | awk 'NR>1{print $1}' | xargs -r scancel
rm -rf ~/projects/llmtrial ~/.venvs/consensus ~/.cache/huggingface ~/.cache/hf
# then repeat from Section 2
```

