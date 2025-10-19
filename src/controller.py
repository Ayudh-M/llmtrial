from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
from pydantic import ValidationError
from .schemas import Envelope
from .canonicalize import canonicalize_for_hash
from .utils import to_json, sha256_hex
from .sanitize import repair_envelope
from .pseudocode import validate_and_normalise_pseudocode, PseudocodeValidationError

def _checked(env_dict: Dict[str, Any]) -> Envelope:
    try:
        return Envelope.model_validate(env_dict)
    except ValidationError:
        repaired = repair_envelope(env_dict)
        return Envelope.model_validate(repaired)


def _apply_pseudocode(env: Envelope) -> Envelope:
    fs = env.final_solution
    if fs and isinstance(fs.canonical_text, str) and fs.canonical_text.strip():
        try:
            normalized, final_return = validate_and_normalise_pseudocode(fs.canonical_text)
            fs.canonical_text = normalized
            setattr(fs, "return_value", final_return)
        except PseudocodeValidationError as exc:
            setattr(fs, "return_value", "")
            env.content = env.content or {}
            env.content.setdefault("pseudocode_error", str(exc))
    return env


def _final_return_value(env: Envelope) -> Optional[str]:
    if env.final_solution is None:
        return None
    ret = getattr(env.final_solution, "return_value", None)
    if isinstance(ret, str) and ret.strip():
        return ret.strip()
    text = env.final_solution.canonical_text or ""
    return text.strip() or None

def _canon_and_hash(text: str, kind: Optional[str]) -> Tuple[str, str]:
    c = canonicalize_for_hash(text or "", kind)
    return c, sha256_hex(c)

def _handshake_accept(prev_env: Envelope, curr_env: Envelope, kind: Optional[str]) -> Optional[str]:
    """If curr_env is a SOLVED+ACCEPT that copies prev_env's canonical result, return canonical; else None."""
    try:
        if curr_env.tag == "[SOLVED]" and curr_env.status == "SOLVED" and curr_env.content and            str(curr_env.content.get("verdict", "")).upper() == "ACCEPT" and curr_env.final_solution and prev_env.final_solution:
            prev_return = _final_return_value(prev_env) or ""
            curr_return = _final_return_value(curr_env) or ""
            ca = canonicalize_for_hash(prev_return, kind)
            cb = canonicalize_for_hash(curr_return, kind)
            if ca == cb and ca != "":
                return ca
    except Exception:
        pass
    return None

def run_controller(task: str, agent_a, agent_b, max_rounds: int = 8, kind: Optional[str] = None) -> Dict[str, Any]:
    transcript: List[Dict[str, Any]] = []
    last_a: Optional[Envelope] = None
    last_b: Optional[Envelope] = None

    for r in range(1, max_rounds + 1):
        # Agent A step
        env_a_raw, _ = agent_a.step(task, transcript)
        env_a = _apply_pseudocode(_checked(env_a_raw))
        transcript.append({"r": r, "actor": "a", "envelope": env_a.model_dump()})

        # Agent B step
        env_b_raw, _ = agent_b.step(task, transcript)
        env_b = _apply_pseudocode(_checked(env_b_raw))
        transcript.append({"r": r, "actor": "b", "envelope": env_b.model_dump()})

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
        final_a = _final_return_value(env_a)
        final_b = _final_return_value(env_b)
        if final_a and final_b:
            ca, ha = _canon_and_hash(final_a, kind)
            cb, hb = _canon_and_hash(final_b, kind)
            if ca == cb and ca != "":
                return {"status": "CONSENSUS", "rounds": r, "canonical_text": ca, "sha256": ha, "transcript": transcript}

        last_a, last_b = env_a, env_b

    return {"status": "NO_CONSENSUS", "rounds": max_rounds, "transcript": transcript}
