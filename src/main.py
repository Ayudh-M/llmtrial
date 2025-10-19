from __future__ import annotations

from src.controller import run_controller
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, List

# --- Package-safe imports: work when invoked as `python -m src.main`
try:
    from .template_loader import get_scenario, load_roleset, load_strategy
    from .strategies import build_strategy
    from .model_loader import load_model_and_tokenizer
    from .agents_hf import HFChatAgent
    from .agents_mock import MockAgent
    from .controller import run_controller
except ImportError:
    # Fallback if someone runs `python src/main.py` by mistake
    from template_loader import get_scenario, load_roleset, load_strategy  # type: ignore
    from strategies import build_strategy  # type: ignore
    from model_loader import load_model_and_tokenizer

def _maybe_shared_loader(model_id_a, model_id_b, dtype):
    """If A and B model IDs match, load once and share; else load separately."""
    tok_a=model_a=tok_b=model_b=None
    if model_id_a==model_id_b:
        tok_a, model_a = load_model_and_tokenizer(model_id_a, dtype=dtype)
        tok_b, model_b = tok_a, model_a
    else:
        ((tok_a,model_a),(tok_b,model_b)) = _maybe_shared_loader(models['a'], models['b'], dtype=use_dtype)
    return (tok_a,model_a),(tok_b,model_b)
  # type: ignore
    from agents_hf import HFChatAgent  # type: ignore
    from agents_mock import MockAgent  # type: ignore
    from controller import run_controller  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
RUNS.mkdir(parents=True, exist_ok=True)


def _now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _dump_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _pick(
    scenario: Dict[str, Any], key: str, fallback_block: str | None = None, subkey: str | None = None
) -> Optional[str]:
    """
    Helper to pick values from scenario:
      - direct key (e.g., "model_a")
      - or from a nested block (e.g., "models": {"a": "...", "b": "..."})
    """
    if key in scenario and scenario[key]:
        return scenario[key]
    if fallback_block and subkey:
        block = scenario.get(fallback_block) or {}
        if isinstance(block, dict) and subkey in block:
            return block[subkey]
    return None
def _norm_number(s: str) -> str:
    s = s.strip()
    # normalize "+4", "4.0" -> "4" ; "-0.0" -> "0"
    try:
        # Avoid float issues by decimal only when needed
        from decimal import Decimal, InvalidOperation, localcontext
        with localcontext() as ctx:
            ctx.prec = 50
            d = Decimal(s)
            if d == d.to_integral():
                return str(d.to_integral())  # integer
            # strip trailing zeros in fractional part
            t = format(d.normalize(), 'f')
            return t
    except Exception:
        return s

def _norm(kind: str|None, s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    if kind == "number":
        return _norm_number(s)
    # default: trim
    return s.strip()

def _handshake_consensus(result: dict, kind: str|None):
    """
    Post-process transcript to detect 'review handshake' consensus:
    - Any SOLVED/REVISED from one agent is a proposal.
    - If the peer replies with SOLVED and (verdict==ACCEPT OR normalized number matches),
      we finalize consensus using the latest agreed normalized canonical_text.
    """
    tx = result.get("transcript") or []
    last_proposal = None  # dict with keys: actor, norm, raw
    agreed = None

    def extract_norm(env):
        fs = (env or {}).get("final_solution") or {}
        ct = fs.get("canonical_text")
        return _norm(kind, ct or ""), ct or ""

    for m in tx:
        env = (m.get("envelope") or {})
        st  = env.get("status","").upper()
        actor = m.get("actor")
        norm, raw = extract_norm(env)

        if st in ("REVISED","SOLVED"):
            # treat as a proposal
            last_proposal = {"actor": actor, "norm": norm, "raw": raw, "env": env, "t": m.get("t")}
            continue

        # If it's not a proposal, skip
        # (We only react on SOLVED from the peer below)
        pass

        # (kept for clarity; not used)
    # second pass: look for SOLVED reviews that accept last proposal
    last_proposal = None
    for m in tx:
        env = (m.get("envelope") or {})
        st  = env.get("status","").upper()
        actor = m.get("actor")
        norm, raw = extract_norm(env)
        verdict = (env.get("content") or {}).get("verdict","").upper()

        if st in ("REVISED","SOLVED"):
            # proposal step
            last_proposal = {"actor": actor, "norm": norm, "raw": raw, "env": env, "t": m.get("t")}
            continue

        if st == "SOLVED" and last_proposal and actor != last_proposal["actor"]:
            # review step
            if verdict == "ACCEPT":
                agreed = last_proposal
                break
            # numeric equivalence fallback
            if _norm(kind, raw) == last_proposal["norm"]:
                agreed = last_proposal
                break

    if agreed:
        # upgrade result to CONSENSUS
        canon = agreed["norm"]
        import hashlib
        sha = hashlib.sha256(canon.encode("utf-8")).hexdigest()
        result["status"] = "CONSENSUS"
        result["canonical_text"] = canon
        result["sha256"] = sha
        result["rounds"] = max([m.get("t",0) for m in tx] or [1])
        # Write finals for A/B in the same normalized form
        for ab in ("final_a","final_b"):
            fin = result.get(ab) or {}
            fs = fin.get("final_solution") or {}
            if fs:
                fs["canonical_text"] = _norm(kind, fs.get("canonical_text",""))
                fs["sha256"] = hashlib.sha256(fs["canonical_text"].encode("utf-8")).hexdigest()
                fin["final_solution"] = fs
                result[ab] = fin
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Two-agent consensus runner")
    ap.add_argument("--scenario", required=True, help="Scenario id from prompts/registry.yaml")
    ap.add_argument("--mock", action="store_true", help="Use deterministic mock agents (no models needed)")
    ap.add_argument("--model-a", dest="model_a", default=None, help="Override model A repo-id")
    ap.add_argument("--model-b", dest="model_b", default=None, help="Override model B repo-id")
    ap.add_argument("--dtype", default=None, help="Model dtype: bf16|fp16|fp32 (defaults come from scenario)")
    args = ap.parse_args()

    # 1) Load scenario + strategy
    try:
        scenario = get_scenario(args.scenario)
    except Exception as e:
        print(f"[error] Could not load scenario '{args.scenario}': {e}", file=sys.stderr)
        sys.exit(2)

    try:
        strat_cfg = load_strategy(scenario["strategy"])
    except KeyError:
        print("[error] Scenario is missing the 'strategy' field.", file=sys.stderr)
        sys.exit(2)
    strategy = build_strategy(strat_cfg)

    task_text: str = scenario.get("task", "")
    if not isinstance(task_text, str) or not task_text.strip():
        print("[error] Scenario 'task' is empty.", file=sys.stderr)
        sys.exit(2)

    kind: Optional[str] = scenario.get("kind")

    # 2) Run path selection
    if args.mock:
        # Pure local smoke test: no rolesets/models required
        mock_solution = scenario.get("mock_solution", "TRUE")
        agent_a = MockAgent("A", solution_text=mock_solution)
        agent_b = MockAgent("B", solution_text=mock_solution)
        result = run_controller(
            task_text,
            agent_a,
            agent_b,
            max_rounds=strategy.max_rounds,
            kind=kind,
            strategy=strategy,
        )
    else:
        # Real models path: load roleset + models
        roleset_path = scenario.get("roleset")
        if not roleset_path:
            print("[error] Scenario is missing 'roleset' for non-mock runs.", file=sys.stderr)
            sys.exit(2)

        try:
            roleset = load_roleset(roleset_path)
        except Exception as e:
            print(f"[error] Could not load roleset '{roleset_path}': {e}", file=sys.stderr)
            sys.exit(2)

        # Model selections
        model_a = args.model_a or _pick(scenario, "model_a", fallback_block="models", subkey="a")
        model_b = args.model_b or _pick(scenario, "model_b", fallback_block="models", subkey="b")
        if not model_a or not model_b:
            print("[error] Missing model ids (model_a/model_b or models:{a,b}) in scenario and no CLI override.", file=sys.stderr)
            sys.exit(2)

        dtype = args.dtype or scenario.get("dtype")

        # Load models
        try:
            tok_a, mdl_a = load_model_and_tokenizer(model_a, dtype=dtype)
            tok_b, mdl_b = load_model_and_tokenizer(model_b, dtype=dtype)
        except Exception as e:
            print(f"[error] Failed to load models/tokenizers: {e}", file=sys.stderr)
            sys.exit(3)

        # Build agents
        try:
            sys_a = roleset["agent_a"]["system"]
            sys_b = roleset["agent_b"]["system"]
        except KeyError:
            print("[error] Roleset must have agent_a.system and agent_b.system", file=sys.stderr)
            sys.exit(2)

        name_a = roleset.get("agent_a", {}).get("name", "Agent A")
        name_b = roleset.get("agent_b", {}).get("name", "Agent B")

        agent_a = HFChatAgent(name_a, sys_a, tok_a, mdl_a, strategy)
        agent_b = HFChatAgent(name_b, sys_b, tok_b, mdl_b, strategy)

        # Controller
        result = run_controller(
            task_text,
            agent_a,
            agent_b,
            max_rounds=strategy.max_rounds,
            kind=kind,
            strategy=strategy,
        )

    # 3) Persist artifact
    out_prefix = f"{_now_stamp()}_{args.scenario}"
    out_path = RUNS / f"{out_prefix}.json"
    out_path.write_text(_dump_json(result), encoding="utf-8")

    # 4) Console summary
    final = result.get("canonical_text")
    if final:
        print("\n=== FINAL TEXT ===\n" + str(final))
    else:
        print("\n(No consensus)")


if __name__ == "__main__":
    # Ensure `src` works as a package when run via `python -m src.main`
    # If you plan to ever run this file directly, add an empty `src/__init__.py` too.
    main()
