from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:  # Package-relative imports when executed as `python -m src.main`
    from .agents_hf import HFChatAgent
    from .agents_mock import MockAgent
    from .controller import run_controller
    from .dsl import extension_from_config
    from .logger import RunMetadata, record_run
    from .model_loader import load_model_and_tokenizer
    from .schemas import get_envelope_validator
    from .strategies import Strategy, build_strategy, list_strategy_ids
    from .template_loader import get_scenario, load_roleset
except ImportError:  # pragma: no cover - fallback for direct execution
    from agents_hf import HFChatAgent  # type: ignore
    from agents_mock import MockAgent  # type: ignore
    from controller import run_controller  # type: ignore
    from dsl import extension_from_config  # type: ignore
    from logger import RunMetadata, record_run  # type: ignore
    from model_loader import load_model_and_tokenizer  # type: ignore
    from schemas import get_envelope_validator  # type: ignore
    from strategies import Strategy, build_strategy, list_strategy_ids  # type: ignore
    from template_loader import get_scenario, load_roleset  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = ROOT / "logs"
DEFAULT_CSV = LOG_DIR / "runs.csv"
DEFAULT_JSONL = LOG_DIR / "runs.jsonl"


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _dump_json(obj: Mapping[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _pick(
    scenario: Mapping[str, Any],
    key: str,
    fallback_block: Optional[str] = None,
    subkey: Optional[str] = None,
) -> Optional[str]:
    if key in scenario and scenario[key]:
        return str(scenario[key])
    if fallback_block and subkey:
        block = scenario.get(fallback_block)
        if isinstance(block, Mapping) and subkey in block and block[subkey]:
            return str(block[subkey])
    return None


def _resolve_strategy_ids(args: argparse.Namespace, scenario: Mapping[str, Any]) -> List[str]:
    if args.all_strategies and args.strategy:
        raise SystemExit("Cannot combine --strategy with --all-strategies.")

    if args.all_strategies:
        return list(list_strategy_ids())

    if args.strategy:
        return [args.strategy]

    strategy_id = scenario.get("strategy")
    if not isinstance(strategy_id, str) or not strategy_id.strip():
        raise SystemExit("Scenario is missing a default strategy and no override was provided.")
    return [strategy_id]


def _strategy_overrides(scenario: Mapping[str, Any]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    block = scenario.get("strategy_overrides")
    if isinstance(block, Mapping):
        overrides.update(block)  # type: ignore[arg-type]
    for key in ("decoding", "max_rounds", "consensus_mode", "validator", "validator_params"):
        if key in scenario and key not in overrides:
            overrides[key] = scenario[key]
    return overrides


def _load_models(
    model_a: str,
    model_b: str,
    *,
    dtype: Optional[str],
) -> Tuple[Tuple[Any, Any], Tuple[Any, Any]]:
    if model_a == model_b:
        tokenizer, model = load_model_and_tokenizer(model_a, dtype=dtype)
        return (tokenizer, model), (tokenizer, model)

    tok_a, mdl_a = load_model_and_tokenizer(model_a, dtype=dtype)
    tok_b, mdl_b = load_model_and_tokenizer(model_b, dtype=dtype)
    return (tok_a, mdl_a), (tok_b, mdl_b)


def _build_agents(
    roleset: Mapping[str, Any],
    tokenizer_pair: Tuple[Any, Any],
    model_pair: Tuple[Any, Any],
    strategy: Strategy,
) -> Tuple[HFChatAgent, HFChatAgent]:
    agent_a_cfg = roleset.get("agent_a") or {}
    agent_b_cfg = roleset.get("agent_b") or {}
    system_a = agent_a_cfg.get("system")
    system_b = agent_b_cfg.get("system")
    if not system_a or not system_b:
        raise SystemExit("Roleset must define system prompts for agent_a and agent_b.")

    name_a = agent_a_cfg.get("name", "Agent A")
    name_b = agent_b_cfg.get("name", "Agent B")

    tok_a, tok_b = tokenizer_pair
    mdl_a, mdl_b = model_pair
    agent_a = HFChatAgent(str(name_a), str(system_a), tok_a, mdl_a, strategy)
    agent_b = HFChatAgent(str(name_b), str(system_b), tok_b, mdl_b, strategy)
    return agent_a, agent_b


def _ensure_roleset_meta(roleset: Mapping[str, Any]) -> str:
    meta = roleset.get("meta") or {}
    if isinstance(meta, Mapping):
        kind = meta.get("task_kind")
        if isinstance(kind, str):
            return kind
    return "unknown"


def _prepare_validators(scenario: Mapping[str, Any]):
    schema_ref = (
        scenario.get("json_envelope_schema")
        or scenario.get("schema")
        or scenario.get("envelope_schema")
    )
    schema_validator = get_envelope_validator(schema_ref) if schema_ref else None

    dsl_config = scenario.get("dsl")
    dsl_validator = None
    if dsl_config:
        extension = extension_from_config(dsl_config)
        dsl_validator = extension.create_validator()

    return dsl_validator, schema_validator


def _mock_agents(strategy: Strategy, mock_solution: str) -> Tuple[MockAgent, MockAgent]:
    agent_a = MockAgent("Mock-A", mock_solution, strategy=strategy)
    agent_b = MockAgent("Mock-B", mock_solution, strategy=strategy)
    return agent_a, agent_b


def _run_once(
    scenario_id: str,
    scenario: Mapping[str, Any],
    strategy_id: str,
    *,
    roleset: Mapping[str, Any],
    strategy: Strategy,
    agent_pair: Tuple[Any, Any],
    kind: Optional[str],
    dsl_validator,
    schema_validator,
    csv_path: Path,
    jsonl_path: Path,
    model_a: str,
    model_b: str,
    extra_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    task_text = str(scenario.get("task", "")).strip()
    if not task_text:
        raise SystemExit("Scenario task description is empty.")

    agent_a, agent_b = agent_pair
    result = run_controller(
        task_text,
        agent_a,
        agent_b,
        max_rounds=strategy.max_rounds,
        kind=kind,
        dsl_validator=dsl_validator,
        schema_validator=schema_validator,
        strategy=strategy,
    )

    roleset_path = str(scenario.get("roleset", ""))
    metadata = RunMetadata(
        scenario_id=scenario_id,
        roleset=roleset_path,
        strategy_id=strategy_id,
        model_a=model_a,
        model_b=model_b,
        extra=dict(extra_meta),
    )

    record_run(result, metadata, csv_path=csv_path, jsonl_path=jsonl_path)
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Two-agent orchestration runner")
    parser.add_argument("--scenario", required=True, help="Scenario id from prompts/registry.yaml")
    parser.add_argument("--strategy", help="Override strategy id (default uses scenario entry)")
    parser.add_argument(
        "--all-strategies",
        action="store_true",
        help="Run the scenario with every registered strategy",
    )
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock agents")
    parser.add_argument("--model-a", dest="model_a", help="Override model id for agent A")
    parser.add_argument("--model-b", dest="model_b", help="Override model id for agent B")
    parser.add_argument("--dtype", help="Model dtype override (bf16, fp16, fp32)")
    parser.add_argument("--csv-log", default=str(DEFAULT_CSV), help="Path to the summary CSV log")
    parser.add_argument("--jsonl-log", default=str(DEFAULT_JSONL), help="Path to the raw JSONL log")
    args = parser.parse_args(argv)

    try:
        scenario = get_scenario(args.scenario)
    except Exception as exc:  # pragma: no cover - configuration errors
        print(f"[error] Failed to load scenario '{args.scenario}': {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    strategy_ids = _resolve_strategy_ids(args, scenario)
    strategy_overrides = _strategy_overrides(scenario)

    if not scenario.get("roleset"):
        print("[error] Scenario must declare a roleset path.", file=sys.stderr)
        raise SystemExit(2)

    try:
        roleset = load_roleset(str(scenario["roleset"]))
    except Exception as exc:  # pragma: no cover - configuration errors
        print(f"[error] Could not load roleset '{scenario['roleset']}': {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    task_kind = _ensure_roleset_meta(roleset)
    kind = scenario.get("kind")
    dsl_validator, schema_validator = _prepare_validators(scenario)

    if args.mock:
        model_a_id = model_b_id = "mock"
        tokenizer_pair = (None, None)
        model_pair = (None, None)
    else:
        model_a_id = args.model_a or _pick(scenario, "model_a", "models", "a")
        model_b_id = args.model_b or _pick(scenario, "model_b", "models", "b")
        if not model_a_id or not model_b_id:
            raise SystemExit(
                "Scenario must define models (model_a/model_b or models:{a,b}) when not using --mock."
            )
        dtype = args.dtype or scenario.get("dtype")
        (tok_a, mdl_a), (tok_b, mdl_b) = _load_models(model_a_id, model_b_id, dtype=dtype)
        tokenizer_pair = (tok_a, tok_b)
        model_pair = (mdl_a, mdl_b)

    csv_path = Path(args.csv_log)
    jsonl_path = Path(args.jsonl_log)

    results: List[Tuple[str, Dict[str, Any]]] = []
    for strategy_id in strategy_ids:
        strategy = build_strategy(strategy_id, overrides=strategy_overrides)

        if args.mock:
            mock_solution = str(scenario.get("mock_solution", "TRUE"))
            agent_pair = _mock_agents(strategy, mock_solution)
        else:
            agent_pair = _build_agents(roleset, tokenizer_pair, model_pair, strategy)

        result = _run_once(
            args.scenario,
            scenario,
            strategy_id,
            roleset=roleset,
            strategy=strategy,
            agent_pair=agent_pair,
            kind=kind,
            dsl_validator=dsl_validator,
            schema_validator=schema_validator,
            csv_path=csv_path,
            jsonl_path=jsonl_path,
            model_a=model_a_id,
            model_b=model_b_id,
            extra_meta={"task_kind": task_kind},
        )

        out_name = f"{_timestamp()}_{args.scenario}_{strategy_id}.json"
        out_path = RUNS_DIR / out_name
        out_path.write_text(_dump_json(result), encoding="utf-8")

        summary = result.get("canonical_text") or "<no consensus>"
        print(f"[{strategy_id}] status={result.get('status')} canonical={summary}")
        results.append((strategy_id, result))

    if len(results) > 1:
        wins = sum(1 for _, res in results if res.get("status") == "CONSENSUS")
        print(f"\nCompleted {len(results)} runs: {wins} reached consensus.")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()

