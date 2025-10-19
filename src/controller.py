from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List, Callable
from pydantic import ValidationError
from .schemas import Envelope
from .canonicalize import canonicalize_for_hash
from .utils import to_json, sha256_hex
from .sanitize import repair_envelope
from .strategies import Strategy
from .validators import get_validator

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

def _prepare_text_turn(obj: Any, validator: Optional[Callable[[str], str]]) -> Dict[str, Any]:
    if isinstance(obj, dict):
        text = obj.get("text") or obj.get("message") or obj.get("content") or ""
        final = obj.get("final_solution") or obj.get("final_answer")
        meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    else:
        text = str(obj or "")
        final = None
        meta = {}

    if validator:
        try:
            text = validator(text)
        except ValueError:
            text = text.strip()
    else:
        text = text.strip()

    final_text = ""
    if isinstance(final, dict):
        final_text = str(final.get("canonical_text") or final.get("text") or "").strip()
    elif isinstance(final, str):
        final_text = final.strip()

    fs = {"canonical_text": final_text} if final_text else None
    payload = {"text": text}
    if fs:
        payload["final_solution"] = fs
    if meta:
        payload["meta"] = meta
    return payload


def run_controller(
    task: str,
    agent_a,
    agent_b,
    max_rounds: int = 8,
    kind: Optional[str] = None,
    strategy: Optional[Strategy] = None,
) -> Dict[str, Any]:
    transcript: List[Dict[str, Any]] = []
    text_mode = bool(strategy and not strategy.envelope_required)
    validator: Optional[Callable[[str], str]] = None
    if text_mode and strategy and strategy.validator_id:
        validator = get_validator(strategy.validator_id, strategy.validator_params)

    last_a: Optional[Any] = None
    last_b: Optional[Any] = None

    for r in range(1, max_rounds + 1):
        # Agent A step
        env_a_raw, _ = agent_a.step(task, transcript)
        if text_mode:
            env_a = _prepare_text_turn(env_a_raw, validator)
            transcript.append({"r": r, "actor": "a", "envelope": env_a})
        else:
            env_a = _checked(env_a_raw)
            transcript.append({"r": r, "actor": "a", "envelope": env_a.model_dump()})

        # Agent B step
        env_b_raw, _ = agent_b.step(task, transcript)
        if text_mode:
            env_b = _prepare_text_turn(env_b_raw, validator)
            transcript.append({"r": r, "actor": "b", "envelope": env_b})
        else:
            env_b = _checked(env_b_raw)
            transcript.append({"r": r, "actor": "b", "envelope": env_b.model_dump()})

        if text_mode:
            final_a = (env_a.get("final_solution") or {}).get("canonical_text") if isinstance(env_a, dict) else None
            final_b = (env_b.get("final_solution") or {}).get("canonical_text") if isinstance(env_b, dict) else None
            if final_a and final_b:
                ca, ha = _canon_and_hash(final_a, kind)
                cb, hb = _canon_and_hash(final_b, kind)
                if ca == cb and ca != "":
                    return {
                        "status": "CONSENSUS",
                        "rounds": r,
                        "canonical_text": ca,
                        "sha256": ha,
                        "transcript": transcript,
                    }
        else:
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
