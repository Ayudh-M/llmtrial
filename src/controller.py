from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import inspect
from functools import lru_cache

from pydantic import ValidationError

from .schemas import Envelope
from .canonicalize import canonicalize_for_hash
from .utils import to_json, sha256_hex
from .sanitize import repair_envelope
from .strategies import Strategy


@lru_cache(None)
def _step_accepts_preparation(agent_cls) -> bool:
    try:
        sig = inspect.signature(agent_cls.step)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == "preparation" and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


def _strategy_for(agent) -> Strategy:
    strat = getattr(agent, "strategy", None)
    return strat if isinstance(strat, Strategy) else Strategy(name=getattr(agent, "name", "agent"))


def _call_step(agent, task: str, transcript: List[Dict[str, Any]], preparation: Dict[str, Any]):
    if _step_accepts_preparation(type(agent)):
        return agent.step(task, transcript, preparation=preparation)
    return agent.step(task, transcript)

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

    strategy_a = _strategy_for(agent_a)
    strategy_b = _strategy_for(agent_b)
    eff_max_rounds = min(max_rounds, strategy_a.max_rounds or max_rounds, strategy_b.max_rounds or max_rounds)

    for r in range(1, eff_max_rounds + 1):
        # Agent A step
        prep_a = strategy_a.prepare_prompt(
            task,
            transcript,
            actor="a",
            agent_name=getattr(agent_a, "name", "agent_a"),
        )
        env_a_raw, raw_a = _call_step(agent_a, task, transcript, prep_a)
        env_a_checked = _checked(env_a_raw)
        validation_a = strategy_a.validate_message(
            env_a_checked,
            raw=raw_a,
            original=env_a_raw,
            transcript=transcript,
            actor="a",
            agent_name=getattr(agent_a, "name", "agent_a"),
        )
        env_a_processed, post_meta_a = strategy_a.postprocess(
            env_a_checked,
            raw=raw_a,
            validation=validation_a,
            transcript=transcript,
            actor="a",
            agent_name=getattr(agent_a, "name", "agent_a"),
        )
        env_a = env_a_processed if isinstance(env_a_processed, Envelope) else _checked(env_a_processed)
        stop_a, reason_a = strategy_a.should_stop(
            env_a,
            validation=validation_a,
            transcript=transcript,
            actor="a",
            agent_name=getattr(agent_a, "name", "agent_a"),
        )
        transcript.append(
            {
                "r": r,
                "actor": "a",
                "envelope": env_a.model_dump(),
                "raw": raw_a,
                "strategy": {
                    "name": strategy_a.name,
                    "max_rounds": strategy_a.max_rounds,
                    "decoding": strategy_a.decoding,
                    "metadata": getattr(strategy_a, "metadata", {}),
                    "hooks": {
                        "prepare": prep_a,
                        "validation": validation_a,
                        "postprocess": post_meta_a,
                        "should_stop": {"stop": stop_a, "reason": reason_a},
                    },
                },
            }
        )

        # Agent B step
        prep_b = strategy_b.prepare_prompt(
            task,
            transcript,
            actor="b",
            agent_name=getattr(agent_b, "name", "agent_b"),
        )
        env_b_raw, raw_b = _call_step(agent_b, task, transcript, prep_b)
        env_b_checked = _checked(env_b_raw)
        validation_b = strategy_b.validate_message(
            env_b_checked,
            raw=raw_b,
            original=env_b_raw,
            transcript=transcript,
            actor="b",
            agent_name=getattr(agent_b, "name", "agent_b"),
        )
        env_b_processed, post_meta_b = strategy_b.postprocess(
            env_b_checked,
            raw=raw_b,
            validation=validation_b,
            transcript=transcript,
            actor="b",
            agent_name=getattr(agent_b, "name", "agent_b"),
        )
        env_b = env_b_processed if isinstance(env_b_processed, Envelope) else _checked(env_b_processed)
        stop_b, reason_b = strategy_b.should_stop(
            env_b,
            validation=validation_b,
            transcript=transcript,
            actor="b",
            agent_name=getattr(agent_b, "name", "agent_b"),
        )
        transcript.append(
            {
                "r": r,
                "actor": "b",
                "envelope": env_b.model_dump(),
                "raw": raw_b,
                "strategy": {
                    "name": strategy_b.name,
                    "max_rounds": strategy_b.max_rounds,
                    "decoding": strategy_b.decoding,
                    "metadata": getattr(strategy_b, "metadata", {}),
                    "hooks": {
                        "prepare": prep_b,
                        "validation": validation_b,
                        "postprocess": post_meta_b,
                        "should_stop": {"stop": stop_b, "reason": reason_b},
                    },
                },
            }
        )

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
