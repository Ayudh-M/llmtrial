from __future__ import annotations

"""Entry point for running two agents under a selected scenario."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - exercised via `python -m src.main`
    from .agents_hf import HFChatAgent
    from .agents_mock import MockAgent
    from .controller import run_controller
    from .model_loader import MISTRAL_MODEL_ID, TINY_MODEL_ID, load_causal_lm
    from .strategies import Strategy, build_strategy
    from .template_loader import get_scenario, load_roleset
except ImportError:  # pragma: no cover - fallback when executed as script
    from agents_hf import HFChatAgent  # type: ignore
    from agents_mock import MockAgent  # type: ignore
    from controller import run_controller  # type: ignore
    from model_loader import MISTRAL_MODEL_ID, TINY_MODEL_ID, load_causal_lm  # type: ignore
    from strategies import Strategy, build_strategy  # type: ignore
    from template_loader import get_scenario, load_roleset  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "runs"
DEFAULT_RUN_DIR.mkdir(parents=True, exist_ok=True)


def _now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _dump_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _pick(
    scenario: Mapping[str, Any],
    key: str,
    *,
    fallback_block: Optional[str] = None,
    subkey: Optional[str] = None,
) -> Optional[str]:
    if key in scenario and scenario[key]:
        value = scenario[key]
        return str(value)
    if fallback_block and subkey:
        block = scenario.get(fallback_block)
        if isinstance(block, Mapping) and block.get(subkey):
            return str(block[subkey])
    return None


def _strategy_from_scenario(scenario: Mapping[str, Any]) -> Strategy:
    raw_strategy = scenario.get("strategy")
    if not raw_strategy:
        raise KeyError("Scenario is missing the 'strategy' identifier.")

    overrides: Dict[str, Any] = {}
    scenario_overrides = scenario.get("strategy_overrides")
    if isinstance(scenario_overrides, Mapping):
        overrides.update(dict(scenario_overrides))

    for key in ("decoding", "max_rounds", "consensus_mode"):
        if key in scenario and key not in overrides:
            overrides[key] = scenario[key]

    return build_strategy(raw_strategy, overrides=overrides)


def _resolve_schema_argument(strategy: Strategy) -> Any:
    schema = strategy.metadata.get("schema") if isinstance(strategy.metadata, Mapping) else None
    if isinstance(schema, str):
        path = (ROOT / schema).resolve()
        return path if path.exists() else schema
    return schema


def _ensure_output_dir(path: Optional[str]) -> Path:
    if not path:
        return DEFAULT_RUN_DIR
    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def _load_model_pair(
    model_a: str,
    model_b: str,
    *,
    dtype: Optional[str],
    device_map: Optional[str],
) -> Tuple[Tuple[Any, Any], Tuple[Any, Any]]:
    load_kwargs: Dict[str, Any] = {}
    if dtype is not None:
        load_kwargs["dtype"] = dtype
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    tok_a, mdl_a = load_causal_lm(model_a, **load_kwargs)
    if model_a == model_b:
        return (tok_a, mdl_a), (tok_a, mdl_a)
    tok_b, mdl_b = load_causal_lm(model_b, **load_kwargs)
    return (tok_a, mdl_a), (tok_b, mdl_b)


def _mock_agents(answer: str, strategy: Strategy) -> Tuple[MockAgent, MockAgent]:
    agent_a = MockAgent("Agent A", answer, strategy=strategy)
    agent_b = MockAgent("Agent B", answer, strategy=strategy)
    return agent_a, agent_b


def _real_agents(
    roleset: Mapping[str, Any],
    model_pair: Tuple[Tuple[Any, Any], Tuple[Any, Any]],
    strategy: Strategy,
) -> Tuple[HFChatAgent, HFChatAgent]:
    tok_a, mdl_a = model_pair[0]
    tok_b, mdl_b = model_pair[1]

    agent_a_cfg = roleset.get("agent_a") if isinstance(roleset, Mapping) else {}
    agent_b_cfg = roleset.get("agent_b") if isinstance(roleset, Mapping) else {}

    try:
        system_a = agent_a_cfg["system"]
        system_b = agent_b_cfg["system"]
    except KeyError as exc:
        missing = "agent_a.system" if "system" not in agent_a_cfg else "agent_b.system"
        raise KeyError(f"Roleset is missing required field '{missing}'.") from exc

    name_a = agent_a_cfg.get("name", "Agent A")
    name_b = agent_b_cfg.get("name", "Agent B")

    agent_a = HFChatAgent(str(name_a), str(system_a), tok_a, mdl_a, strategy)
    agent_b = HFChatAgent(str(name_b), str(system_b), tok_b, mdl_b, strategy)
    return agent_a, agent_b


def _scenario_task(scenario: Mapping[str, Any]) -> str:
    task = scenario.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Scenario 'task' must be a non-empty string.")
    return task.strip()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run a two-agent consensus scenario")
    parser.add_argument("--scenario", required=True, help="Scenario id from prompts/registry.yaml")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock agents instead of HF models")
    parser.add_argument("--model-a", dest="model_a", help="Override scenario model id for agent A")
    parser.add_argument("--model-b", dest="model_b", help="Override scenario model id for agent B")
    parser.add_argument("--dtype", help="Model dtype override (bf16|fp16|fp32)")
    parser.add_argument("--device-map", help="Transformers device_map override (e.g., auto, cpu)")
    parser.add_argument(
        "--preset",
        choices=("tiny", "mistral"),
        help="Quick model override: 'tiny' uses the TinyStories stub, 'mistral' selects Mistral-7B",
    )
    parser.add_argument("--output-dir", help="Directory for run artifacts (defaults to ./runs)")

    args = parser.parse_args(argv)

    try:
        scenario = get_scenario(args.scenario)
    except Exception as exc:  # pragma: no cover - defensive for CLI usage
        print(f"[error] Failed to load scenario '{args.scenario}': {exc}", file=sys.stderr)
        sys.exit(2)

    try:
        strategy = _strategy_from_scenario(scenario)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[error] Could not prepare strategy: {exc}", file=sys.stderr)
        sys.exit(2)

    try:
        task_text = _scenario_task(scenario)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(2)

    schema_arg = _resolve_schema_argument(strategy)
    controller_kwargs: Dict[str, Any] = {"max_rounds": strategy.max_rounds}
    if schema_arg is not None:
        controller_kwargs["schema_validator"] = schema_arg

    if args.mock:
        answer = str(scenario.get("mock_solution", "42"))
        agent_a, agent_b = _mock_agents(answer, strategy)
    else:
        roleset_path = scenario.get("roleset")
        if not roleset_path:
            print("[error] Scenario is missing the 'roleset' path required for HF runs.", file=sys.stderr)
            sys.exit(2)

        try:
            roleset = load_roleset(str(roleset_path))
        except Exception as exc:  # pragma: no cover - scenario validation
            print(f"[error] Failed to load roleset '{roleset_path}': {exc}", file=sys.stderr)
            sys.exit(2)

        if args.preset == "tiny":
            model_a_id = model_b_id = TINY_MODEL_ID
        elif args.preset == "mistral":
            model_a_id = model_b_id = MISTRAL_MODEL_ID
        else:
            model_a_id = args.model_a or _pick(scenario, "model_a", fallback_block="models", subkey="a")
            model_b_id = args.model_b or _pick(scenario, "model_b", fallback_block="models", subkey="b")

        if not model_a_id or not model_b_id:
            print(
                "[error] Scenario must define model_a/model_b (or models:{a,b}) or use --preset/--model overrides.",
                file=sys.stderr,
            )
            sys.exit(2)

        dtype = args.dtype or scenario.get("dtype")
        device_map = args.device_map or scenario.get("device_map")

        try:
            model_pair = _load_model_pair(model_a_id, model_b_id, dtype=dtype, device_map=device_map)
        except Exception as exc:  # pragma: no cover - real model loading
            print(f"[error] Failed to load Hugging Face models: {exc}", file=sys.stderr)
            sys.exit(3)

        try:
            agent_a, agent_b = _real_agents(roleset, model_pair, strategy)
        except Exception as exc:  # pragma: no cover - roleset validation
            print(f"[error] Could not initialise agents: {exc}", file=sys.stderr)
            sys.exit(2)

    result = run_controller(task_text, agent_a, agent_b, **controller_kwargs)

    output_dir = _ensure_output_dir(args.output_dir)
    out_path = output_dir / f"{_now_stamp()}_{args.scenario}.json"
    out_path.write_text(_dump_json(result), encoding="utf-8")

    final_text = result.get("canonical_text")
    status = result.get("status", "NO_CONSENSUS")
    print(f"[status] {status}")
    if final_text:
        print("\n=== FINAL TEXT ===\n" + str(final_text))
    else:
        print("\n(No consensus reached)")
    print(f"[saved] {out_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
