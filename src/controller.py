from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
from pydantic import ValidationError

from .schemas import Envelope
from .canonicalize import canonicalize_for_hash
from .utils import sha256_hex
from .sanitize import repair_envelope
from .dsl import DSLValidator, DSLValidationError, DSLParseResult


def _checked(
    env_dict: Dict[str, Any],
    dsl_validator: Optional[DSLValidator],
) -> Tuple[Envelope, Dict[str, Any], Optional[DSLParseResult]]:
    candidate = dict(env_dict) if isinstance(env_dict, dict) else {}
    try:
        env = Envelope.model_validate(candidate)
    except ValidationError:
        repaired = repair_envelope(candidate)
        env = Envelope.model_validate(repaired)
        candidate = repaired
    else:
        candidate = dict(candidate)

    parsed: Optional[DSLParseResult] = None
    if dsl_validator is not None:
        try:
            parsed = dsl_validator.validate(candidate)
        except DSLValidationError as exc:
            if exc.envelope is None:
                exc.envelope = candidate
            raise
    return env, dict(candidate), parsed


def _canon_and_hash(text: str, kind: Optional[str]) -> Tuple[str, str]:
    c = canonicalize_for_hash(text or "", kind)
    return c, sha256_hex(c)


def _handshake_accept(prev_env: Envelope, curr_env: Envelope, kind: Optional[str]) -> Optional[str]:
    """If curr_env is a SOLVED+ACCEPT that copies prev_env's canonical_text, return canonical; else None."""
    try:
        if (
            curr_env.tag == "[SOLVED]"
            and curr_env.status == "SOLVED"
            and curr_env.content
            and str(curr_env.content.get("verdict", "")).upper() == "ACCEPT"
            and curr_env.final_solution
            and prev_env.final_solution
        ):
            ca = canonicalize_for_hash(prev_env.final_solution.canonical_text, kind)
            cb = canonicalize_for_hash(curr_env.final_solution.canonical_text, kind)
            if ca == cb and ca != "":
                return ca
    except Exception:
        pass
    return None


def run_controller(
    task: str,
    agent_a,
    agent_b,
    max_rounds: int = 8,
    kind: Optional[str] = None,
    dsl_validator: Optional[DSLValidator] = None,
) -> Dict[str, Any]:
    transcript: List[Dict[str, Any]] = []
    dsl_trace: List[Dict[str, Any]] = []
    last_a: Optional[Envelope] = None
    last_b: Optional[Envelope] = None

    for r in range(1, max_rounds + 1):
        # Agent A step
        env_a_raw, _ = agent_a.step(task, transcript)
        try:
            env_a, env_a_data, parse_a = _checked(env_a_raw, dsl_validator)
        except DSLValidationError as err:
            envelope = err.envelope if isinstance(err.envelope, dict) else {}
            transcript.append({"r": r, "actor": "a", "envelope": envelope, "errors": list(err.errors)})
            if dsl_validator is not None:
                dsl_trace.append({
                    "round": r,
                    "actor": "a",
                    "errors": list(err.errors),
                    "grammar_sha256": dsl_validator.grammar_sha256,
                })
            return {
                "status": "INVALID_DSL",
                "rounds": r,
                "offender": "a",
                "errors": list(err.errors),
                "transcript": transcript,
                "dsl_trace": dsl_trace,
                "final_message": None,
            }
        transcript.append({"r": r, "actor": "a", "envelope": env_a_data})
        if parse_a is not None:
            dsl_trace.append(parse_a.to_trace_entry(r, "a"))

        # Agent B step
        env_b_raw, _ = agent_b.step(task, transcript)
        try:
            env_b, env_b_data, parse_b = _checked(env_b_raw, dsl_validator)
        except DSLValidationError as err:
            envelope = err.envelope if isinstance(err.envelope, dict) else {}
            transcript.append({"r": r, "actor": "b", "envelope": envelope, "errors": list(err.errors)})
            if dsl_validator is not None:
                dsl_trace.append({
                    "round": r,
                    "actor": "b",
                    "errors": list(err.errors),
                    "grammar_sha256": dsl_validator.grammar_sha256,
                })
            return {
                "status": "INVALID_DSL",
                "rounds": r,
                "offender": "b",
                "errors": list(err.errors),
                "transcript": transcript,
                "dsl_trace": dsl_trace,
                "final_message": None,
            }
        transcript.append({"r": r, "actor": "b", "envelope": env_b_data})
        if parse_b is not None:
            dsl_trace.append(parse_b.to_trace_entry(r, "b"))

        # Handshake acceptance paths (either direction)
        if last_b:
            canon = _handshake_accept(last_b, env_a, kind)
            if canon is not None:
                c, h = _canon_and_hash(canon, kind)
                final_message = None
                if parse_a is not None:
                    final_message = {
                        "actor": "a",
                        "round": r,
                        "envelope": env_a_data,
                        "dsl": parse_a.to_trace_entry(r, "a"),
                    }
                return {
                    "status": "CONSENSUS",
                    "rounds": r,
                    "canonical_text": c,
                    "sha256": h,
                    "transcript": transcript,
                    "dsl_trace": dsl_trace,
                    "final_message": final_message,
                }
        if last_a:
            canon = _handshake_accept(last_a, env_b, kind)
            if canon is not None:
                c, h = _canon_and_hash(canon, kind)
                final_message = None
                if parse_b is not None:
                    final_message = {
                        "actor": "b",
                        "round": r,
                        "envelope": env_b_data,
                        "dsl": parse_b.to_trace_entry(r, "b"),
                    }
                return {
                    "status": "CONSENSUS",
                    "rounds": r,
                    "canonical_text": c,
                    "sha256": h,
                    "transcript": transcript,
                    "dsl_trace": dsl_trace,
                    "final_message": final_message,
                }

        # Equality fallback if both produced final solutions this round
        final_a = (
            env_a.final_solution.canonical_text
            if (env_a.final_solution and env_a.final_solution.canonical_text)
            else None
        )
        final_b = (
            env_b.final_solution.canonical_text
            if (env_b.final_solution and env_b.final_solution.canonical_text)
            else None
        )
        if final_a and final_b:
            ca, ha = _canon_and_hash(final_a, kind)
            cb, hb = _canon_and_hash(final_b, kind)
            if ca == cb and ca != "":
                final_message = None
                if parse_b is not None:
                    final_message = {
                        "actor": "b",
                        "round": r,
                        "envelope": env_b_data,
                        "dsl": parse_b.to_trace_entry(r, "b"),
                    }
                elif parse_a is not None:
                    final_message = {
                        "actor": "a",
                        "round": r,
                        "envelope": env_a_data,
                        "dsl": parse_a.to_trace_entry(r, "a"),
                    }
                return {
                    "status": "CONSENSUS",
                    "rounds": r,
                    "canonical_text": ca,
                    "sha256": ha,
                    "transcript": transcript,
                    "dsl_trace": dsl_trace,
                    "final_message": final_message,
                }

        last_a, last_b = env_a, env_b

    return {
        "status": "NO_CONSENSUS",
        "rounds": max_rounds,
        "transcript": transcript,
        "dsl_trace": dsl_trace,
        "final_message": None,
    }
