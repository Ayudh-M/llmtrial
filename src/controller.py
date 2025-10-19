from __future__ import annotations
from collections import Counter
from dataclasses import asdict
from typing import Dict, Any, Tuple, Optional, List
from pydantic import ValidationError
from .schemas import Envelope
from .canonicalize import canonicalize_for_hash
from .utils import sha256_hex, parse_acl_message, ACLParseError
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

def _parse_intent(actor_label: str, env: Envelope):
    content = env.content or {}
    if content is None:
        return None
    if not isinstance(content, dict):
        raise ValueError(f"Agent {actor_label} content must be an object with an 'acl' field.")
    acl = content.get("acl")
    if acl is None:
        return None
    try:
        return parse_acl_message(acl)
    except ACLParseError as exc:
        raise ValueError(f"Invalid ACL message from agent {actor_label}: {exc}") from exc


def _intent_summary(intent_counts: Dict[str, Counter]) -> Dict[str, Any]:
    return {"intent_counts": {actor: dict(counter) for actor, counter in intent_counts.items()}}

def run_controller(task: str, agent_a, agent_b, max_rounds: int = 8, kind: Optional[str] = None) -> Dict[str, Any]:
    transcript: List[Dict[str, Any]] = []
    solved_text: Dict[str, Optional[str]] = {"a": None, "b": None}
    intent_counts: Dict[str, Counter] = {"a": Counter(), "b": Counter()}

    def maybe_finish(round_idx: int) -> Optional[Dict[str, Any]]:
        text_a = solved_text.get("a")
        text_b = solved_text.get("b")
        if text_a and text_b:
            ca, ha = _canon_and_hash(text_a, kind)
            cb, hb = _canon_and_hash(text_b, kind)
            if ca == cb and ca != "":
                return {
                    "status": "CONSENSUS",
                    "rounds": round_idx,
                    "canonical_text": ca,
                    "sha256": ha,
                    "transcript": transcript,
                    "analytics": _intent_summary(intent_counts),
                }
        return None

    for r in range(1, max_rounds + 1):
        # Agent A step
        env_a_raw, _ = agent_a.step(task, transcript)
        env_a = _checked(env_a_raw)
        parsed_a = _parse_intent("A", env_a)
        if parsed_a:
            intent_counts["a"][parsed_a.intent] += 1
        transcript.append({
            "r": r,
            "actor": "a",
            "intent": parsed_a.intent if parsed_a else None,
            "acl": asdict(parsed_a) if parsed_a else None,
            "envelope": env_a.model_dump(),
        })
        if env_a.is_solved() and parsed_a and parsed_a.intent == "SOLVED":
            solved_text["a"] = env_a.final_solution.canonical_text or ""
        else:
            solved_text["a"] = None

        maybe = maybe_finish(r)
        if maybe:
            return maybe

        # Agent B step
        env_b_raw, _ = agent_b.step(task, transcript)
        env_b = _checked(env_b_raw)
        parsed_b = _parse_intent("B", env_b)
        if parsed_b:
            intent_counts["b"][parsed_b.intent] += 1
        transcript.append({
            "r": r,
            "actor": "b",
            "intent": parsed_b.intent if parsed_b else None,
            "acl": asdict(parsed_b) if parsed_b else None,
            "envelope": env_b.model_dump(),
        })
        if env_b.is_solved() and parsed_b and parsed_b.intent == "SOLVED":
            solved_text["b"] = env_b.final_solution.canonical_text or ""
        else:
            solved_text["b"] = None

        maybe = maybe_finish(r)
        if maybe:
            return maybe

    return {
        "status": "NO_CONSENSUS",
        "rounds": max_rounds,
        "transcript": transcript,
        "analytics": _intent_summary(intent_counts),
    }
