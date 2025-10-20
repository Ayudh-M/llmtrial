Files included:
- src/model_loader.py
- src/agents_hf.py
- src/main.py  (adds --hf path while keeping --mock)
- run_gpu_hf.job (GPU job script for Snellius)
- run_gpu.job    (mock GPU script for quick checks)
- requirements.txt

Usage (Snellius):
  module purge
  module load 2025
  module spider Python
  module load Python/3.11.6-GCCcore-13.3.0   # pick one listed by the spider command
  source ~/.venvs/consensus/bin/activate
  mkdir -p runs logs

  # Real HF run with Mistral-7B on both agents using a registered scenario
  sbatch run_gpu_hf.job mistral_math_smoke

  # Observe
  #  jid=$(sbatch ... | awk '{print $4}')
  #  while [ ! -f logs/consensus-hf-$jid.out ]; do sleep 5; done
  #  tail -f logs/consensus-hf-$jid.out

  # After completion
  latest=$(ls -1dt runs/*/ | head -n 1)
  sed -n '1,120p' "$latest/final.json"
  tail -n 5 runs/diagnostics.csv
