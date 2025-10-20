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
# List available Python builds in the 2025 stack and pick one that exists.
module spider Python
# Example (adjust to one listed by the spider command):
module load Python/3.11.6-GCCcore-13.3.0

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

## 5) H100 job script (already in the repo)

The repository ships an H100-friendly wrapper `run_gpu_mistral.job`. Review it to confirm the
partition/account values match your project, then make sure it is executable:

```bash
chmod +x run_gpu_mistral.job
```

No here-doc copy/paste is required, so you avoid the truncated script issue that happens when the
closing `SLURM` marker is misplaced.

---

## 6) Submit a GPU run (H100)

```bash
JID=$(sbatch run_gpu_mistral.job mistral_math_smoke | awk '{print $4}'); echo "JOBID=$JID"
while [ ! -f "logs/consensus_mistral-$JID.out" ]; do sleep 5; done
tail -f "logs/consensus_mistral-$JID.out"
```

---

## 7) Inspect the latest run result

**Option A (helper script in repo):**

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
JID=$(sbatch run_gpu_mistral.job regex_email_basic | awk '{print $4}'); echo "JOBID=$JID"
while [ ! -f "logs/consensus_mistral-$JID.out" ]; do sleep 5; done
tail -f "logs/consensus_mistral-$JID.out"
```

You can override model/dtype in `src.main` if needed (e.g., `--model-a`, `--model-b`, `--dtype`), but by default the **strategy and models** come from the registry.

---

## 9) Common gotchas

* **Do not paste angle brackets** in commands (e.g., `<your-username>`). Use real values or variables as shown.
* **Stay inside your workspace** when cleaning up: never run `rm -rf -- *` from `$HOME/projects`. Only delete `"$LLMTRIAL_WORKDIR/llmtrial"` as shown above.
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

