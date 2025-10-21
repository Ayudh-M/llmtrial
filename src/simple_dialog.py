# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import List, Tuple

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


def build_system(role_text: str, strategy_text: str) -> str:
    return f"{role_text}\n\n{strategy_text}".strip()


def run_dialog(
    scenario_text: str,
    strategy_id: str,
    roleset_id: str,
    turns: int,
    model_a: str,
    model_b: str,
    gen_cfg: GenConfig,
    seed: int | None,
    outdir: str = "logs",
) -> Tuple[List[TurnRecord], RunSummary]:
    os.makedirs(outdir, exist_ok=True)
    strategy = STRATEGIES[strategy_id]
    roles = ROLESETS[roleset_id]
    sys_a = build_system(roles["A"], strategy)
    sys_b = build_system(roles["B"], strategy)

    seed_everything(seed)
    agent_a = SimpleHF(model_a)
    agent_b = SimpleHF(model_b)

    transcript: List[TurnRecord] = []
    msg = scenario_text.strip()
    tot_in = tot_out = 0

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
        scenario=scenario_text[:160],
        strategy=strategy_id,
        roleset=roleset_id,
        turns=turns,
        model_a=model_a,
        model_b=model_b,
        seed=seed,
        total_prompt_tokens=tot_in,
        total_output_tokens=tot_out,
    )

    ts = time.strftime("%Y%m%d-%H%M%S")
    base = f"runs_fixed_{strategy_id}_{roleset_id}_{ts}"
    jsonl_path = os.path.join(outdir, f"{base}.jsonl")
    csv_path = os.path.join(outdir, f"{base}.csv")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "config": asdict(summary),
                    "transcript": [asdict(t) for t in transcript],
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
    return transcript, summary


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
        scenario_text=scenario_text,
        strategy_id=args.strategy,
        roleset_id=args.roleset,
        turns=args.turns,
        model_a=args.model_a,
        model_b=args.model_b,
        gen_cfg=cfg,
        seed=args.seed,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
