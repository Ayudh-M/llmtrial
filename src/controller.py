from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List

from .strategies import Strategy
from pydantic import ValidationError
from .schemas import Envelope
from .canonicalize import canonicalize_for_hash
from .utils import to_json, sha256_hex
from .sanitize import repair_envelope

def _checked(env_dict: Dict[str, Any]) -> Envelope:
    try:
        return Envelope.model_validate(env_dict)
    except ValidationError:
        repaired = repair_envelope(env_dict)
        return Envelope.model_validate(repaired)

def _canon_and_hash(text: str, kind: Optional[str]) -> Tuple[str, str]:
    c = canonicalize_for_hash(text or "", kind)
    return c, sha256_hex(c)

def _handshake_accept(prev_env: Envelope, curr_env: Envelope, kind: Optional[str]) -> Optional[str]:
    """If curr_env is a SOLVED+ACCEPT that copies prev_env's canonical_text, return canonical; else None."""
    try:
        if curr_env.tag == "[SOLVED]" and curr_env.status == "SOLVED" and curr_env.content and            str(curr_env.content.get("verdict", "")).upper() == "ACCEPT" and curr_env.final_solution and prev_env.final_solution:
            ca = canonicalize_for_hash(prev_env.final_solution.canonical_text, kind)
            cb = canonicalize_for_hash(curr_env.final_solution.canonical_text, kind)
            if ca == cb and ca != "":
                return ca
    except Exception:
        pass
    return None

def run_controller(task: str, agent_a, agent_b, max_rounds: int = 8, kind: Optional[str] = None) -> Dict[str, Any]:
    transcript: List[Dict[str, Any]] = []
    last_a: Optional[Envelope] = None
    last_b: Optional[Envelope] = None
    strategy: Optional[Strategy] = None

    strat_a = getattr(agent_a, "strategy", None)
    strat_b = getattr(agent_b, "strategy", None)
    if strat_a is not None and strat_a is strat_b:
        strategy = strat_a

    controller_ctx: Dict[str, Any] = {"task": task, "kind": kind, "transcript": transcript}

    for r in range(1, max_rounds + 1):
        if strategy:
            controller_ctx["round"] = r
            controller_ctx["transcript"] = transcript
            strategy.apply_pre_round_hooks(controller_ctx)

        # Agent A step
        env_a_raw, _ = agent_a.step(task, transcript)
        env_a = _checked(env_a_raw)
        transcript.append({"r": r, "actor": "a", "envelope": env_a.model_dump()})

        # Agent B step
        env_b_raw, _ = agent_b.step(task, transcript)
        env_b = _checked(env_b_raw)
        transcript.append({"r": r, "actor": "b", "envelope": env_b.model_dump()})

        if strategy:
            controller_ctx["transcript"] = transcript
            strategy.apply_controller_behaviors(controller_ctx)

        # Handshake acceptance paths (either direction)
        if last_b:
            canon = _handshake_accept(last_b, env_a, kind)
            if canon is not None:
                c, h = _canon_and_hash(canon, kind)
                return {"status": "CONSENSUS", "rounds": r, "canonical_text": c, "sha256": h, "transcript": transcript}
        if last_a:
            canon = _handshake_accept(last_a, env_b, kind)
            if canon is not None:
                c, h = _canon_and_hash(canon, kind)
                return {"status": "CONSENSUS", "rounds": r, "canonical_text": c, "sha256": h, "transcript": transcript}

        # Equality fallback if both produced final solutions this round
        final_a = env_a.final_solution.canonical_text if (env_a.final_solution and env_a.final_solution.canonical_text) else None
        final_b = env_b.final_solution.canonical_text if (env_b.final_solution and env_b.final_solution.canonical_text) else None
        if final_a and final_b:
            ca, ha = _canon_and_hash(final_a, kind)
            cb, hb = _canon_and_hash(final_b, kind)
            if ca == cb and ca != "":
                return {"status": "CONSENSUS", "rounds": r, "canonical_text": ca, "sha256": ha, "transcript": transcript}

        last_a, last_b = env_a, env_b

    return {"status": "NO_CONSENSUS", "rounds": max_rounds, "transcript": transcript}
