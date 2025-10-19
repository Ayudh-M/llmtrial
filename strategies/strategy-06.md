# Strategy 1 — Dual‑agent JSON consensus for Mistral (Instruct‑tuned preferred)

## Why this strategy fits Mistral
- Mistral base models (e.g., `Mistral-7B-v0.1`) are **pretrained** text generators; they are not optimized for instruction following. Instruct variants (e.g., `Mistral-7B-Instruct-v0.2`) are **fine‑tuned** to follow chat/instructions and respect control tokens like `[INST] ... [/INST]` (via chat templates). Use an **Instruct** model when possible.
- The model always emits a **text stream** (tokens). It can produce prose or JSON or anything you prompt it to—unless you constrain decoding. We therefore (a) prime it with a strict JSON‑only protocol and (b) optionally **constrain decoding** to a JSON schema using grammar/structured output backends (e.g., TGI Guidance, vLLM guided_json, or Outlines).
- **Chain‑of‑thought** is just intermediate natural‑language tokens the model *might* emit if asked (e.g., “let’s think step by step”). We forbid it and accept **only** the compact JSON envelope. Not serializing CoT doesn’t change model weights; it simply prevents leaking reasoning steps and keeps outputs parseable.

## Controller expectations (summarized)
- Each agent returns exactly one JSON object per turn (no prose outside JSON).
- Agents may set tags in `public_message`:
  - `[CONTACT]` when they need the peer **this turn** → the controller routes anyway, but this tag clarifies intent.
  - `[SOLVED]` when their `final_solution.canonical_text` is complete and they set `status:"SOLVED"`.
- The run halts only when **both** agents emit `status:"SOLVED"` **and** their `final_solution.canonical_text` are **identical after normalization** (whitespace normalization + newline folding). The controller enforces this.
- The `status` field is not magic—it's a **convention** we instruct the model to follow. Your Python code validates it and can also apply safety nets (e.g., if `[CONTACT]` is present but status isn’t `NEED_PEER`, override to `NEED_PEER`).

## Prompting & decoding notes for Mistral
- Prefer an **Instruct** model (e.g., `mistralai/Mistral-7B-Instruct-v0.2`) and use the tokenizer’s **chat template** to wrap system/user content. If a chat template is present, build prompts via `tokenizer.apply_chat_template([...], add_generation_prompt=True)`; otherwise fall back to a plain string prompt.
- For structured output, choose one:
  1) **Greedy decoding** (no sampling) with a strict prompt, plus JSON extraction fallback.
  2) **Grammar‑guided decoding** to a JSON schema (TGI “Guidance” JSON grammar, vLLM `guided_json`, or Outlines) for higher compliance.
- Keep `temperature` low for structure; allow small `top_p`/`temperature` only if creativity is required **inside** JSON fields.

## JSON envelope (identical for both agents)
{
  "role": "<Your role name>",
  "domain": "<short domain label>",
  "task_understanding": "<one concise sentence>",
  "public_message": "<one or two short lines; may contain [CONTACT] or [SOLVED]>",
  "artifact": {
    "type": "<component_spec|code_patch|outline|fact_pack|source_pack|plan|dataset|results>",
    "content": { }
  },
  "needs_from_peer": ["<0..3 concrete asks>"],
  "handoff_to": "<peer role name>",
  "status": "WORKING | NEED_PEER | PROPOSED | READY_TO_SOLVE | SOLVED",
  "final_solution": {
    "canonical_text": "<required when SOLVED; the complete final text>"
  }
}

## Micro‑rules the agents must follow
1) **When asking for help** — set `status:"NEED_PEER"`, include `[CONTACT]` in `public_message`, list ≤3 crisp asks in `needs_from_peer`.
2) **When proposing** — set `status:"PROPOSED"` and put the candidate in `artifact.content`.
3) **When ready** — set `status:"READY_TO_SOLVE"` and include the full candidate; invite adoption.
4) **When final** — set `status:"SOLVED"`, include `[SOLVED]`, and place the full canonical solution in `final_solution.canonical_text`. If adopting your peer’s text, copy it **verbatim**.
5) Never emit `[SOLVED]` without a complete `final_solution.canonical_text`.

## Role pack guidance (append per agent)
- **UI/UX Designer** — produce `component_spec` with: `name`, `elements[]`, `states[]`, `constraints[]`, `accessibility[]`, `data_contract{inputs,outputs}`.
- **Programmer** — produce `code_patch` with: `language`, `files[{path,content}]`, `run_cmd`, `tests[{name,input,expected}]`.
- **Writer** — produce `outline` → `results` (final prose) with: title, sections[], tone, reading level, word target.
- **Physicist / Biologist** — produce `fact_pack` with: definitions[], laws/mechanisms[], approximations/assumptions[], references[].

## Controller safety nets (recommended)
- If `public_message` contains `[CONTACT]` but `status` ≠ `NEED_PEER`, coerce `status="NEED_PEER"`.
- If both agents are `READY_TO_SOLVE` with identical candidates, auto‑promote to `SOLVED` by copying canonical text into both envelopes (optional).
- If JSON parsing fails, send a structured error envelope back to the peer and continue.
