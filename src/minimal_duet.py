from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

if __package__:
    from .model_loader import load_model_and_tokenizer
    from .template_loader import get_scenario, load_roleset
else:  # pragma: no cover - fallback for direct execution
    from model_loader import load_model_and_tokenizer  # type: ignore
    from template_loader import get_scenario, load_roleset  # type: ignore


TURN_COUNT = 8


@dataclass
class RunConfig:
    dataset: str
    language: str
    pair: str
    rep: int
    model_a: Optional[str]
    model_b: Optional[str]
    max_new_tokens: int
    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    do_sample: Optional[bool]
    dtype: Optional[str]
    mock: bool
    logdir: Path


@dataclass
class TurnResult:
    turn: int
    actor: str
    raw: Optional[str]
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    model_id: Optional[str]
    started_at: str
    finished_at: str
    max_new_tokens: int
    error: Optional[str] = None


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret boolean value: {value}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal duet runner without intent parsing")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--logdir", default="logs/minimal_duet")
    parser.add_argument("--model-a", dest="model_a")
    parser.add_argument("--model-b", dest="model_b")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", dest="top_p", type=float)
    parser.add_argument("--top-k", dest="top_k", type=int)
    parser.add_argument("--do-sample", dest="do_sample", nargs="?", const=True, type=_parse_bool)
    parser.add_argument("--dtype")
    parser.add_argument("--mock", action="store_true", help="Run without loading models, emitting mock replies")
    return parser


def _resolve_config(args: argparse.Namespace) -> RunConfig:
    logdir = Path(args.logdir).resolve()
    logdir.mkdir(parents=True, exist_ok=True)

    do_sample: Optional[bool]
    if args.do_sample is None:
        do_sample = None
    else:
        do_sample = bool(args.do_sample)

    return RunConfig(
        dataset=str(args.dataset),
        language=str(args.language),
        pair=str(args.pair),
        rep=int(args.rep),
        model_a=str(args.model_a) if args.model_a else None,
        model_b=str(args.model_b) if args.model_b else None,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature) if args.temperature is not None else None,
        top_p=float(args.top_p) if args.top_p is not None else None,
        top_k=int(args.top_k) if args.top_k is not None else None,
        do_sample=do_sample,
        dtype=str(args.dtype) if args.dtype else None,
        mock=bool(args.mock),
        logdir=logdir,
    )


def _scenario_id(cfg: RunConfig) -> str:
    return f"{cfg.dataset}:{cfg.language}:{cfg.pair}:rep={cfg.rep}"


def _model_label(cfg: RunConfig, actor: str) -> Optional[str]:
    if actor == "A":
        return cfg.model_a or ("mock-A" if cfg.mock else None)
    return cfg.model_b or ("mock-B" if cfg.mock else None)


def _load_prompts(cfg: RunConfig) -> Tuple[str, str, str]:
    scenario_id = _scenario_id(cfg)
    scenario = get_scenario(scenario_id)

    task_text = scenario.get("task")
    if not isinstance(task_text, str) or not task_text:
        raise SystemExit(f"Scenario '{scenario_id}' is missing a task text.")

    roleset_path = scenario.get("roleset")
    if not isinstance(roleset_path, str) or not roleset_path:
        raise SystemExit(f"Scenario '{scenario_id}' is missing a roleset reference.")

    roleset = load_roleset(str(roleset_path))
    system_a = roleset.get("agent_a", {}).get("system")
    system_b = roleset.get("agent_b", {}).get("system")
    if not isinstance(system_a, str) or not isinstance(system_b, str):
        raise SystemExit(f"Roleset '{roleset_path}' must define system prompts for agent_a and agent_b.")

    return system_a, system_b, task_text


def _normalize_tokenizer(tokenizer: Any) -> Any:
    if (
        getattr(tokenizer, "pad_token_id", None) is None
        and getattr(tokenizer, "eos_token_id", None) is not None
    ):
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def _prepare_models(cfg: RunConfig) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
    if cfg.mock:
        return None, None, None, None

    if not cfg.model_a or not cfg.model_b:
        raise SystemExit("--model-a and --model-b are required unless --mock is set.")

    model_a, tokenizer_a = load_model_and_tokenizer(cfg.model_a, dtype=cfg.dtype)
    tokenizer_a = _normalize_tokenizer(tokenizer_a)

    if cfg.model_a == cfg.model_b:
        model_b, tokenizer_b = model_a, tokenizer_a
    else:
        model_b, tokenizer_b = load_model_and_tokenizer(cfg.model_b, dtype=cfg.dtype)
        tokenizer_b = _normalize_tokenizer(tokenizer_b)

    return model_a, tokenizer_a, model_b, tokenizer_b


def _ensure_attention_mask(tokenizer: Any, input_ids: torch.Tensor) -> torch.Tensor:
    _normalize_tokenizer(tokenizer)
    return torch.ones_like(input_ids)


def _build_inputs(tokenizer: Any, system_prompt: str, user_prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            if isinstance(input_ids, Mapping):
                input_ids = input_ids["input_ids"]
            attention_mask = _ensure_attention_mask(tokenizer, input_ids)
            return input_ids, attention_mask
        except Exception:
            pass

    formatted = f"{system_prompt}\n\n{user_prompt}"
    encoded = tokenizer(
        formatted,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = _ensure_attention_mask(tokenizer, input_ids)
    return input_ids, attention_mask


@torch.inference_mode()
def _generate(
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    cfg: RunConfig,
) -> Tuple[str, int, int]:
    input_ids, attention_mask = _build_inputs(tokenizer, system_prompt, user_prompt)
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    generate_kwargs: Dict[str, Any] = {
        "max_new_tokens": cfg.max_new_tokens,
        "return_dict_in_generate": True,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if cfg.temperature is not None:
        generate_kwargs["temperature"] = cfg.temperature
    if cfg.top_p is not None:
        generate_kwargs["top_p"] = cfg.top_p
    if cfg.top_k is not None:
        generate_kwargs["top_k"] = cfg.top_k
    if cfg.do_sample is not None:
        generate_kwargs["do_sample"] = cfg.do_sample

    if getattr(tokenizer, "eos_token_id", None) is not None:
        generate_kwargs.setdefault("eos_token_id", tokenizer.eos_token_id)

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generate_kwargs,
    )

    if hasattr(output, "sequences"):
        sequences = output.sequences
    else:
        sequences = output

    prompt_length = input_ids.shape[-1]
    generated_tokens = sequences[0, prompt_length:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return text, int(prompt_length), int(generated_tokens.shape[-1])


def _mock_turn(actor: str, turn_index: int) -> Tuple[str, int, int]:
    text = f"[Mock-{actor} turn{turn_index}]"
    return text, 0, 0


def _append_csv_row(path: Path, row: Mapping[str, Any]) -> None:
    fieldnames = [
        "dataset",
        "language",
        "pair",
        "rep",
        "model_a",
        "model_b",
        "wall_seconds_total",
        "status",
    ]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False)
        handle.write("\n")


def _run_turn(
    actor: str,
    turn_index: int,
    incoming: str,
    system_prompt: str,
    cfg: RunConfig,
    model: Optional[Any],
    tokenizer: Optional[Any],
) -> TurnResult:
    start = time.time()
    start_iso = datetime.fromtimestamp(start, timezone.utc).isoformat()
    try:
        if cfg.mock or model is None or tokenizer is None:
            text, prompt_tokens, completion_tokens = _mock_turn(actor, turn_index)
        else:
            text, prompt_tokens, completion_tokens = _generate(
                model,
                tokenizer,
                system_prompt,
                incoming,
                cfg,
            )
        end = time.time()
        latency = end - start
        return TurnResult(
            turn=turn_index,
            actor=actor,
            raw=text,
            latency_s=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_id=_model_label(cfg, actor),
            started_at=start_iso,
            finished_at=datetime.fromtimestamp(end, timezone.utc).isoformat(),
            max_new_tokens=cfg.max_new_tokens,
        )
    except Exception as exc:  # pragma: no cover - defensive
        end = time.time()
        latency = end - start
        return TurnResult(
            turn=turn_index,
            actor=actor,
            raw=None,
            latency_s=latency,
            prompt_tokens=0,
            completion_tokens=0,
            model_id=_model_label(cfg, actor),
            started_at=start_iso,
            finished_at=datetime.fromtimestamp(end, timezone.utc).isoformat(),
            max_new_tokens=cfg.max_new_tokens,
            error=f"{type(exc).__name__}: {exc}",
        )


def _summarize_turn(turn: TurnResult) -> Dict[str, Any]:
    data = {
        "turn": turn.turn,
        "actor": turn.actor,
        "latency_s": round(turn.latency_s, 6),
        "prompt_tokens": turn.prompt_tokens,
        "completion_tokens": turn.completion_tokens,
        "model_id": turn.model_id,
        "started_at": turn.started_at,
        "finished_at": turn.finished_at,
        "max_new_tokens": turn.max_new_tokens,
    }
    if turn.raw is not None:
        data["raw"] = turn.raw
    if turn.error is not None:
        data["error"] = turn.error
    return data


def _execute(cfg: RunConfig) -> Tuple[str, Dict[str, Any]]:
    system_a, system_b, task_text = _load_prompts(cfg)

    model_a = model_b = tokenizer_a = tokenizer_b = None
    if not cfg.mock:
        model_a, tokenizer_a, model_b, tokenizer_b = _prepare_models(cfg)

    run_start = time.time()
    started_at = datetime.fromtimestamp(run_start, timezone.utc).isoformat()
    turns: List[TurnResult] = []
    status = "OK"
    error_message: Optional[str] = None

    next_for_a = task_text
    next_for_b = ""

    for idx in range(1, TURN_COUNT + 1):
        actor = "A" if idx % 2 == 1 else "B"
        if actor == "A":
            incoming = next_for_a
            result = _run_turn(actor, idx, incoming, system_a, cfg, model_a, tokenizer_a)
            if result.error:
                status = "ERROR"
                error_message = result.error
                turns.append(result)
                break
            next_for_b = result.raw or ""
        else:
            incoming = next_for_b
            result = _run_turn(actor, idx, incoming, system_b, cfg, model_b, tokenizer_b)
            if result.error:
                status = "ERROR"
                error_message = result.error
                turns.append(result)
                break
            next_for_a = result.raw or ""
        turns.append(result)

    run_end = time.time()
    finished_at = datetime.fromtimestamp(run_end, timezone.utc).isoformat()
    wall_seconds = run_end - run_start

    record: Dict[str, Any] = {
        "dataset": cfg.dataset,
        "language": cfg.language,
        "pair": cfg.pair,
        "rep": cfg.rep,
        "scenario_id": _scenario_id(cfg),
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_seconds": round(wall_seconds, 6),
        "turns": [_summarize_turn(t) for t in turns],
        "system_a": system_a,
        "system_b": system_b,
        "task_text": task_text,
        "model_a": _model_label(cfg, "A"),
        "model_b": _model_label(cfg, "B"),
        "status": status,
    }
    if error_message:
        record["error"] = error_message

    return status, record


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    cfg = _resolve_config(args)

    status = "ERROR"
    record: Dict[str, Any] = {}
    csv_row: Dict[str, Any] = {}
    try:
        status, record = _execute(cfg)
    except Exception as exc:  # pragma: no cover - defensive
        status = "ERROR"
        record = {
            "dataset": cfg.dataset,
            "language": cfg.language,
            "pair": cfg.pair,
            "rep": cfg.rep,
            "scenario_id": _scenario_id(cfg),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": 0.0,
            "turns": [],
            "system_a": None,
            "system_b": None,
            "task_text": None,
            "model_a": _model_label(cfg, "A"),
            "model_b": _model_label(cfg, "B"),
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }

    csv_row = {
        "dataset": record.get("dataset"),
        "language": record.get("language"),
        "pair": record.get("pair"),
        "rep": record.get("rep"),
        "model_a": record.get("model_a"),
        "model_b": record.get("model_b"),
        "wall_seconds_total": record.get("wall_seconds"),
        "status": status,
    }

    csv_path = cfg.logdir / "runs.csv"
    jsonl_path = cfg.logdir / "runs.jsonl"

    _append_csv_row(csv_path, csv_row)
    _append_jsonl(jsonl_path, record)

    return 0 if status == "OK" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
