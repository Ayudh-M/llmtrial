# Running on Snellius (GPU)

This repo ships a SLURM script `run_gpu_mistral.job` tuned for Snellius' H100 partition.

## 1) Copy & prepare
```bash
ssh <user>@snellius.surf.nl
export LLMTRIAL_WORKDIR="$HOME/projects/llmtrial_ws"
mkdir -p "$LLMTRIAL_WORKDIR"
cd "$LLMTRIAL_WORKDIR"
git clone <your_repo_url> llmtrial   # or upload this tarball
cd llmtrial
mkdir -p logs runs
```

## 2) Submit a smoke test with Mistral-7B
```bash
JID=$(sbatch run_gpu_mistral.job mistral_math_smoke | awk '{print $4}')
echo "JOBID=$JID"
# or override models explicitly:
# sbatch run_gpu_mistral.job mistralai/Mistral-7B-Instruct-v0.3 mistralai/Mistral-7B-Instruct-v0.3 bf16
```

## 3) Inspect logs and outputs
```bash
squeue -u $USER
while [ ! -f "logs/consensus_mistral-$JID.out" ]; do sleep 5; done
tail -f "logs/consensus_mistral-$JID.out"
ls -lah runs/
```

### Notes
- The job script sets `HF_HOME` (and `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`) to `$TMPDIR/hf` if available (node-local scratch), else `/scratch-shared/$USER/hf`.
- For instruction-tuned models like `mistralai/Mistral-7B-Instruct-v0.3`, the job uses the chat template automatically.
- Before running anything, load the 2025 stack and an available Python module:
  `module purge && module load 2025 && module spider Python`.  When the spider
  output shows a candidate (e.g. `Python/3.11.6-GCCcore-13.3.0`), load the
  listed toolchain first (`module load GCCcore/13.3.0`) before `module load`-ing
  the Python module itself.
- Outputs are written to `runs/<timestamp>_<scenario>.json` with the final envelopes and SHA-256 of the canonical text.

