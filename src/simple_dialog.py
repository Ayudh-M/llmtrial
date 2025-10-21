# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List

from .presets import ROLESETS, STRATEGIES
from .simple_agents import GenConfig, SimpleHF, seed_everything


@dataclass
class TurnRecord:
    r: int
    actor: str
    text_in: str
    text_out: str
    prompt_tokens: int
    output_tokens: int
    stop_reason: str


@dataclass
class RunSummary:
    scenario: str
    strategy: str
    roleset: str
    turns: int
    model_a: str
    model_b: str
    seed: int | None
    total_prompt_tokens: int
    total_output_tokens: int
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool


def build_system(role_text: str, strategy_text: str) -> str:
    return f"{role_text}\n\n{strategy_text}".strip()


def run_dialog(
    scenario_text: str | None = None,
    strategy_id: str | None = None,
    roleset_id: str | None = None,
    turns: int = 6,
    model_a: str | None = None,
    model_b: str | None = None,
    gen_cfg: GenConfig | None = None,
    seed: int | None = None,
    outdir: str = "logs",
    *,
    scenario: str | None = None,
    strategy: str | None = None,
    roleset: str | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> Dict[str, Any]:
    """Run a fixed-turn dialog between two agents.

    Parameters support both the legacy names (``scenario_text``/``strategy_id``/``roleset_id``)
    and the shorter aliases (``scenario``/``strategy``/``roleset``) used by the matrix runner.
    Generation parameters can be supplied either via ``gen_cfg`` (legacy) or directly as
    keyword arguments.
    """
    scenario = scenario if scenario is not None else scenario_text
    strategy = strategy if strategy is not None else strategy_id
    roleset = roleset if roleset is not None else roleset_id

    if scenario is None:
        raise ValueError("scenario text must be provided")
    if strategy is None:
        raise ValueError("strategy id must be provided")
    if roleset is None:
        raise ValueError("roleset id must be provided")
    if model_a is None or model_b is None:
        raise ValueError("model ids must be provided")

    os.makedirs(outdir, exist_ok=True)
    strategy_text = STRATEGIES[strategy]
    roles = ROLESETS[roleset]
    sys_a = build_system(roles["A"], strategy_text)
    sys_b = build_system(roles["B"], strategy_text)

    default_cfg = GenConfig()
    if gen_cfg is None:
        cfg_kwargs = {
            "max_new_tokens": max_new_tokens
            if max_new_tokens is not None
            else default_cfg.max_new_tokens,
            "temperature": temperature
            if temperature is not None
            else default_cfg.temperature,
            "top_p": top_p if top_p is not None else default_cfg.top_p,
            "do_sample": default_cfg.do_sample,
        }
        gen_cfg = GenConfig(**cfg_kwargs)
    else:
        if any(v is not None for v in (max_new_tokens, temperature, top_p)):
            repl_kwargs = {}
            if max_new_tokens is not None:
                repl_kwargs["max_new_tokens"] = max_new_tokens
            if temperature is not None:
                repl_kwargs["temperature"] = temperature
            if top_p is not None:
                repl_kwargs["top_p"] = top_p
            gen_cfg = replace(gen_cfg, **repl_kwargs)

    seed_everything(seed)
    agent_a = SimpleHF(model_a)
    agent_b = SimpleHF(model_b)

    transcript: List[TurnRecord] = []
    msg = scenario.strip()
    tot_in = tot_out = 0
    start = time.time()

    for r in range(1, turns + 1):
        actor = "A" if r % 2 == 1 else "B"
        agent = agent_a if actor == "A" else agent_b
        sys_prompt = sys_a if actor == "A" else sys_b

        out, tok_in, tok_out, stop = agent.respond(sys_prompt, msg, gen_cfg)
        transcript.append(TurnRecord(r, actor, msg, out, tok_in, tok_out, stop))
        tot_in += tok_in
        tot_out += tok_out
        msg = out

    summary = RunSummary(
        scenario=scenario[:160],
        strategy=strategy,
        roleset=roleset,
        turns=turns,
        model_a=model_a,
        model_b=model_b,
        seed=seed,
        total_prompt_tokens=tot_in,
        total_output_tokens=tot_out,
        max_new_tokens=gen_cfg.max_new_tokens,
        temperature=gen_cfg.temperature,
        top_p=gen_cfg.top_p,
        do_sample=gen_cfg.do_sample,
    )

    ts = time.strftime("%Y%m%d-%H%M%S")
    base = f"runs_fixed_{strategy}_{roleset}_{ts}"
    jsonl_path = os.path.join(outdir, f"{base}.jsonl")
    csv_path = os.path.join(outdir, f"{base}.csv")

    elapsed = time.time() - start
    cfg_dict = asdict(summary)
    transcript_dicts = [asdict(t) for t in transcript]

    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "config": cfg_dict,
                    "transcript": transcript_dicts,
                    "elapsed_sec": elapsed,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("r,actor,prompt_tokens,output_tokens,stop_reason\n")
        for t in transcript:
            f.write(
                f"{t.r},{t.actor},{t.prompt_tokens},{t.output_tokens},{t.stop_reason}\n"
            )

    print(f"WROTE {jsonl_path}")
    print(f"WROTE {csv_path}")
    return {
        "config": cfg_dict,
        "transcript": transcript_dicts,
        "out_jsonl": jsonl_path,
        "out_csv": csv_path,
        "elapsed_sec": elapsed,
    }


def main():
    parser = argparse.ArgumentParser("fixed-turn A↔B dialog")
    parser.add_argument("--scenario", required=True, help="Task text or @path/to/file.txt")
    parser.add_argument("--strategy", default="NL", choices=STRATEGIES.keys())
    parser.add_argument("--roleset", default="Planner-Solver", choices=ROLESETS.keys())
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument(
        "--model-a", default="mistralai/Mistral-7B-Instruct-v0.3"
    )
    parser.add_argument(
        "--model-b", default="mistralai/Mistral-7B-Instruct-v0.3"
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="logs")
    args = parser.parse_args()

    scenario_text = args.scenario
    if scenario_text.startswith("@"):
        with open(scenario_text[1:], encoding="utf-8") as f:
            scenario_text = f.read()

    cfg = GenConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
    )

    run_dialog(
        scenario=scenario_text,
        strategy=args.strategy,
        roleset=args.roleset,
        turns=args.turns,
        model_a=args.model_a,
        model_b=args.model_b,
        gen_cfg=cfg,
        seed=args.seed,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
