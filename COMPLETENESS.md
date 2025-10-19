# Completeness Matrix (Tasks × Judges × Strategies)
- Judges implemented: math (format), regex (pos/neg), sql (inline DB), json (parse), boolean, headline (≤12 words), paraphrase (label), mcq_arc (A–D), winogrande (A/B).
- Strategy wiring in code: S01 (greedy, JSON-only), S03 (allow scratch), S05 (self-consistency k=3 scaffold), S06 (arbiter hint on repeated mismatch), S09-ish (per-strategy max_new_tokens).
- Controller: SHA equality, auto-promotion on matching PROPOSED/READY_TO_SOLVE, basic arbiter hint, NEED_PEER forwarding via transcript-aware prompting in Agent.
- Logging: JSONL in runs/results.jsonl with tokens (estimated), rounds, latency, models, strategy.
- Tests: tiny utils/controller smoke tests.
