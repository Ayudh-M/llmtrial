# Fixed-Turn A↔B Dialog Runner

This repository now ships a minimal, robust "pass-through" dialog runner that wires two
Hugging Face causal language models into an alternating conversation for a fixed number
of turns. Strategies and roles are pure prompt presets—there are no control trailers,
consensus loops, or envelope schemas in the execution path.

Legacy orchestrator modules remain in the tree for reference, but the supported workflow
is entirely driven by the new runner described below.

---

## 1. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
mkdir -p logs
```

The runner loads Hugging Face models directly. Ensure you have credentials configured if
you are pulling gated checkpoints.

---

## 2. Prompt presets

Prompt text is defined in `src/presets.py`:

* `STRATEGIES` holds stylistic instructions (e.g., `NL`, `JSON_SCHEMA`, `PSEUDOCODE`).
* `ROLESETS` defines the role-specific guidance for actors `A` and `B`.

These values are pure text snippets that become the system prompt for each actor. Add or
edit entries as needed to change behaviors—no schema changes are required.

---

## 3. Running a dialog

Use the `src.simple_dialog` module to launch a fixed-turn exchange. The CLI accepts
scenario text, strategy/roleset IDs, model identifiers, and generation parameters.

```bash
python -m src.simple_dialog \
  --scenario "Task: Sum 1..100. Output only the number." \
  --strategy JSON_SCHEMA \
  --roleset Planner-Solver \
  --turns 4 \
  --model-a mistralai/Mistral-7B-Instruct-v0.3 \
  --model-b mistralai/Mistral-7B-Instruct-v0.3 \
  --max-new-tokens 128 \
  --temperature 0.2 \
  --top-p 0.95 \
  --seed 7 \
  --outdir logs
```

* `--scenario` accepts raw text or `@path/to/file.txt` to read task text from disk.
* Generation defaults (max tokens, temperature, top-p) can be overridden per run.
* Models `A` and `B` are independent—use different IDs if desired.
* The runner alternates speakers starting with actor `A` and feeds each response as the
  next turn's input message.

---

## 4. Output artifacts

Each run writes two files to the specified `--outdir` (default `logs/`):

1. **Transcript JSONL** – `logs/runs_fixed_<strategy>_<roleset>_<timestamp>.jsonl`
   ```json
   {
     "config": {
       "scenario": "Task: Sum 1..100. Output only the number.",
       "strategy": "JSON_SCHEMA",
       "roleset": "Planner-Solver",
       "turns": 4,
       "model_a": "mistralai/Mistral-7B-Instruct-v0.3",
       "model_b": "mistralai/Mistral-7B-Instruct-v0.3",
       "seed": 7,
       "total_prompt_tokens": 123,
       "total_output_tokens": 456
     },
     "transcript": [
       {
         "r": 1,
         "actor": "A",
         "text_in": "Task: …",
         "text_out": "…",
         "prompt_tokens": 101,
         "output_tokens": 32,
         "stop_reason": "eos_or_sample"
       },
       {
         "r": 2,
         "actor": "B",
         "text_in": "…",
         "text_out": "…",
         "prompt_tokens": 88,
         "output_tokens": 40,
         "stop_reason": "length"
       }
     ]
   }
   ```
2. **Token accounting CSV** – `logs/runs_fixed_<strategy>_<roleset>_<timestamp>.csv`
   ```csv
   r,actor,prompt_tokens,output_tokens,stop_reason
   1,A,101,32,eos_or_sample
   2,B,88,40,length
   ```

The JSONL file holds a single object per run and is ready for offline inspection or
manual evaluation. The CSV provides easy-to-plot token counts per turn.

---

## 5. Manual evaluation workflow

Quantitative judging happens outside this repository. After a run completes, upload the
JSONL transcript (and optionally the CSV) to a stronger evaluation model with a prompt
like:

```
You are an evaluator. Given a strategy, roleset, the original task, and the
turn-ordered transcript (JSON), compute:
- turn_count
- best_turn (earliest turn with correct complete answer)
- final_quality (0–1)
- hallucination_turns (list)
- adherence_score to requested strategy (0–1)
- notes (1-3 sentences)

Return JSON only with those fields.
```

---

## 6. Optional SLURM helper

The repository includes `run_fixed.job`, a ready-to-submit Snellius SLURM script that
launches the fixed runner on a single GPU. Submit it with `sbatch run_fixed.job` to
recreate the example workload used during development. Logs stream to
`logs/llm_fixed-<JOBID>.out`.

---

## 7. Tests

Unit coverage for the new flow lives in `tests/test_simple_dialog.py`. Legacy
control-trailer tests are marked as skipped to reflect the simplified runtime. Run the
active suite with:

```bash
pytest -q
```

The dummy-based test stubs the HF layer, so the suite runs quickly on CPU-only
workstations.
