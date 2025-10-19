from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List, Callable
from collections import Counter
from dataclasses import asdict
from typing import Dict, Any, Tuple, Optional, List

from jsonschema import Draft7Validator
import inspect
from functools import lru_cache


from .strategies import Strategy
from pydantic import ValidationError

from .schemas import Envelope
from .canonicalize import canonicalize_for_hash
from .utils import sha256_hex, parse_acl_message, ACLParseError
from .utils import sha256_hex
from .sanitize import repair_envelope
from .json_enforcer import validate_envelope
from .strategies import Strategy
from .validators import get_validator


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

def _enforce_schema(env: Envelope, validator: Optional[Draft7Validator], actor: str, round_no: int) -> None:
    if not validator:
        return
    ok, errors = validate_envelope(env.model_dump(exclude_none=True), validator)
    if ok:
        return
    details = "; ".join(errors) if errors else "unknown validation error"
    raise ValueError(f"Schema validation failed for {actor} on round {round_no}: {details}")
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
    schema_validator: Optional[Draft7Validator] = None,
) -> Dict[str, Any]:
    strategy: Optional[Strategy] = None,
) -> Dict[str, Any]:
    transcript: List[Dict[str, Any]] = []
    text_mode = bool(strategy and not strategy.envelope_required)
    validator: Optional[Callable[[str], str]] = None
    if text_mode and strategy and strategy.validator_id:
        validator = get_validator(strategy.validator_id, strategy.validator_params)

    last_a: Optional[Any] = None
    last_b: Optional[Any] = None

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
    last_a: Optional[Envelope] = None
    last_b: Optional[Envelope] = None
    strategy: Optional[Strategy] = None

    strat_a = getattr(agent_a, "strategy", None)
    strat_b = getattr(agent_b, "strategy", None)
    if strat_a is not None and strat_a is strat_b:
        strategy = strat_a

    controller_ctx: Dict[str, Any] = {"task": task, "kind": kind, "transcript": transcript}

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
    for r in range(1, max_rounds + 1):
        if strategy:
            controller_ctx["round"] = r
            controller_ctx["transcript"] = transcript
            strategy.apply_pre_round_hooks(controller_ctx)

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
        env_a = _checked(env_a_raw)
        _enforce_schema(env_a, schema_validator, getattr(agent_a, "name", "agent_a"), r)
        transcript.append({"r": r, "actor": "a", "envelope": env_a.model_dump()})
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
        _enforce_schema(env_b, schema_validator, getattr(agent_b, "name", "agent_b"), r)
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
        env_a = _apply_pseudocode(_checked(env_a_raw))
        transcript.append({"r": r, "actor": "a", "envelope": env_a.model_dump()})

        # Agent B step
        env_b_raw, _ = agent_b.step(task, transcript)
        env_b = _apply_pseudocode(_checked(env_b_raw))
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
        final_a = _final_return_value(env_a)
        final_b = _final_return_value(env_b)
        if final_a and final_b:
            ca, ha = _canon_and_hash(final_a, kind)
            cb, hb = _canon_and_hash(final_b, kind)
            if ca == cb and ca != "":
                return {"status": "CONSENSUS", "rounds": r, "canonical_text": ca, "sha256": ha, "transcript": transcript}

        maybe = maybe_finish(r)
        if maybe:
            return maybe

    return {
        "status": "NO_CONSENSUS",
        "rounds": max_rounds,
        "transcript": transcript,
        "analytics": _intent_summary(intent_counts),
    }
