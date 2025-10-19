# Running on Snellius (GPU)

This repo ships a SLURM script `run_gpu_hf.job` tuned for Snellius' `gpu` partition (1/4 node: 18 CPU cores, 1 GPU, ~120 GiB RAM).

## 1) Copy & prepare
```bash
ssh <user>@snellius.surf.nl
git clone <your_repo_url>  # or upload this tarball
cd llm-orchestrator-main
mkdir -p logs runs
```

## 2) Submit a smoke test with Mistral-7B-v0.1
```bash
sbatch run_gpu_hf.job mistral_v01_smoketest
# or override models explicitly:
# sbatch run_gpu_hf.job mistralai/Mistral-7B-v0.1 mistralai/Mistral-7B-v0.1 bfloat16
```

## 3) Inspect logs and outputs
```bash
squeue -u $USER
tail -f logs/consensus-hf-*.out
ls -lah runs/
```

### Notes
- The job script sets `HF_HOME` (and `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`) to `$TMPDIR/hf` if available (node-local scratch), else `/scratch-shared/$USER/hf`.
- For base models like `mistralai/Mistral-7B-v0.1` (no chat template), we fall back to a minimal, safe prompt format.
- Outputs are written to `runs/<timestamp>_<scenario>.json` with the final envelopes and SHA-256 of the canonical text.