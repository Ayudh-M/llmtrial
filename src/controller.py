from __future__ import annotations

import inspect
from collections import Counter
from dataclasses import asdict
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

from jsonschema import Draft7Validator
from pydantic import ValidationError

from .canonicalize import canonicalize_for_hash
from .dsl import DSLParseResult, DSLValidationError, DSLValidator
from .json_enforcer import validate_envelope
from .pseudocode import PseudocodeValidationError, validate_and_normalise_pseudocode
from .sanitize import repair_envelope
from .schemas import Envelope
from .strategies import Strategy
from .utils import ACLParseError, ACLParseResult, ALLOWED_PERFORMATIVES, parse_acl_message, sha256_hex
from .validators import get_validator


@lru_cache(maxsize=None)
def _step_accepts_preparation(agent_cls: type) -> bool:
    try:
        sig = inspect.signature(agent_cls.step)
    except (TypeError, ValueError, AttributeError):
        return False

    for param in sig.parameters.values():
        if param.kind == param.VAR_KEYWORD:
            return True
        if param.name == "preparation" and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


def _strategy_for(agent: Any, override: Optional[Strategy]) -> Strategy:
    if override is not None:
        return override
    strategy = getattr(agent, "strategy", None)
    if isinstance(strategy, Strategy):
        return strategy
    return Strategy(id="default", name=getattr(agent, "name", "agent"))


def _call_step(
    agent: Any,
    task: str,
    transcript: List[Dict[str, Any]],
    preparation: Optional[Dict[str, Any]],
) -> Tuple[Any, Any]:
    if preparation is not None and _step_accepts_preparation(type(agent)):
        return agent.step(task, transcript, preparation=preparation)
    return agent.step(task, transcript)


def _apply_pseudocode(env: Envelope) -> Envelope:
    final = env.final_solution
    if final and isinstance(final.canonical_text, str) and final.canonical_text.strip():
        try:
            normalised, return_value = validate_and_normalise_pseudocode(final.canonical_text)
        except PseudocodeValidationError as exc:
            env.content = env.content or {}
            env.content.setdefault("pseudocode_error", str(exc))
        else:
            final.canonical_text = normalised
            setattr(final, "return_value", return_value)
    return env


def _checked(
    env_candidate: Any,
    dsl_validator: Optional[DSLValidator],
) -> Tuple[Envelope, Dict[str, Any], Optional[DSLParseResult]]:
    candidate = dict(env_candidate) if isinstance(env_candidate, dict) else {}
    try:
        env = Envelope.model_validate(candidate)
    except ValidationError:
        repaired = repair_envelope(candidate)
        env = Envelope.model_validate(repaired)
        candidate = repaired
    env = _apply_pseudocode(env)

    parsed: Optional[DSLParseResult] = None
    if dsl_validator is not None:
        try:
            parsed = dsl_validator.validate(candidate)
        except DSLValidationError as exc:
            if exc.envelope is None:
                exc.envelope = candidate
            raise
    return env, candidate, parsed


def _final_return_value(env: Envelope) -> Optional[str]:
    final = env.final_solution
    if not final:
        return None
    ret = getattr(final, "return_value", None)
    if isinstance(ret, str) and ret.strip():
        return ret.strip()
    text = final.canonical_text or ""
    return text.strip() or None


def _canon_and_hash(text: str, kind: Optional[str]) -> Tuple[str, str]:
    canonical = canonicalize_for_hash(text or "", kind)
    return canonical, sha256_hex(canonical)


def _parse_intent(actor_label: str, env: Envelope) -> Optional[Any]:
    content = env.content or {}
    if not isinstance(content, dict):
        raise ValueError(f"Agent {actor_label} content must be an object with an 'acl' field.")
    acl = content.get("acl")
    if acl is None:
        intent = content.get("intent")
        if isinstance(intent, str) and intent.strip():
            intent_upper = intent.strip().upper()
            if intent_upper not in ALLOWED_PERFORMATIVES:
                raise ValueError(
                    f"Invalid intent '{intent}' from agent {actor_label}. Allowed intents: {', '.join(ALLOWED_PERFORMATIVES)}."
                )
            body = content.get("message") or content.get("notes") or content.get("summary") or intent_upper
            next_action = content.get("next_action") if isinstance(content.get("next_action"), str) else None
            return ACLParseResult(intent=intent_upper, content=str(body), next_action=next_action)
        return None
    try:
        return parse_acl_message(acl)
    except ACLParseError as exc:
        raise ValueError(f"Invalid ACL message from agent {actor_label}: {exc}") from exc


def _intent_summary(intent_counts: Dict[str, Counter]) -> Dict[str, Any]:
    return {"intent_counts": {actor: dict(counter) for actor, counter in intent_counts.items()}}


def _handshake_accept(prev_env: Envelope, curr_env: Envelope, kind: Optional[str]) -> Optional[str]:
    if (
        curr_env.tag == "[SOLVED]"
        and curr_env.status == "SOLVED"
        and curr_env.content
        and str(curr_env.content.get("verdict", "")).upper() == "ACCEPT"
        and prev_env.final_solution
        and curr_env.final_solution
    ):
        prev_return = _final_return_value(prev_env) or ""
        curr_return = _final_return_value(curr_env) or ""
        ca = canonicalize_for_hash(prev_return, kind)
        cb = canonicalize_for_hash(curr_return, kind)
        if ca == cb and ca:
            return ca
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
        text = str(text).strip()

    final_text = ""
    if isinstance(final, dict):
        final_text = str(final.get("canonical_text") or final.get("text") or "").strip()
    elif isinstance(final, str):
        final_text = final.strip()

    payload: Dict[str, Any] = {"text": text}
    if final_text:
        payload["final_solution"] = {"canonical_text": final_text}
    if meta:
        payload["meta"] = meta
    return payload


def run_controller(
    task: str,
    agent_a: Any,
    agent_b: Any,
    max_rounds: int = 8,
    *,
    kind: Optional[str] = None,
    dsl_validator: Optional[DSLValidator] = None,
    schema_validator: Optional[Draft7Validator] = None,
    strategy: Optional[Strategy] = None,
) -> Dict[str, Any]:
    transcript: List[Dict[str, Any]] = []
    dsl_trace: List[Dict[str, Any]] = []
    intent_counts: Dict[str, Counter] = {"a": Counter(), "b": Counter()}

    strat_a = _strategy_for(agent_a, strategy)
    strat_b = _strategy_for(agent_b, strategy)

    controller_ctx: Dict[str, Any] = {"task": task, "kind": kind, "transcript": transcript}
    for strat in (strat_a, strat_b):
        strat.apply_controller_behaviors(controller_ctx)

    eff_max_rounds = min(
        max_rounds,
        strat_a.max_rounds or max_rounds,
        strat_b.max_rounds or max_rounds,
    )

    text_mode = not strat_a.envelope_required or not strat_b.envelope_required
    validator: Optional[Callable[[str], str]] = None
    if text_mode and (strat_a.validator_id or strat_b.validator_id):
        chosen = strat_a if strat_a.validator_id else strat_b
        validator = get_validator(chosen.validator_id, chosen.validator_params)  # type: ignore[arg-type]

    last_env: Dict[str, Optional[Envelope]] = {"a": None, "b": None}
    last_parse: Dict[str, Optional[DSLParseResult]] = {"a": None, "b": None}
    solved_text: Dict[str, Optional[str]] = {"a": None, "b": None}
    final_message: Optional[Dict[str, Any]] = None

    for round_idx in range(1, eff_max_rounds + 1):
        controller_ctx["round"] = round_idx
        controller_ctx["transcript"] = transcript
        strat_a.apply_pre_round_hooks(controller_ctx)
        strat_b.apply_pre_round_hooks(controller_ctx)

        # Agent A step
        prep_a = strat_a.prepare_prompt(
            task,
            transcript,
            actor="a",
            agent_name=getattr(agent_a, "name", "agent_a"),
        )
        env_a_raw, raw_a = _call_step(agent_a, task, transcript, prep_a)

        if text_mode:
            env_a = _prepare_text_turn(env_a_raw, validator)
            transcript.append({
                "r": round_idx,
                "actor": "a",
                "envelope": env_a,
                "raw": raw_a,
                "strategy": {
                    "name": strat_a.name,
                    "max_rounds": strat_a.max_rounds,
                    "decoding": strat_a.decoding,
                    "metadata": strat_a.metadata,
                    "hooks": {"prepare": prep_a},
                },
            })
            final_a = env_a.get("final_solution", {}).get("canonical_text") if isinstance(env_a, dict) else None
            solved_text["a"] = (final_a or "").strip() or None
        else:
            try:
                env_a, env_a_data, parse_a = _checked(env_a_raw, dsl_validator)
            except DSLValidationError as err:
                envelope = err.envelope if isinstance(err.envelope, dict) else {}
                transcript.append({"r": round_idx, "actor": "a", "envelope": envelope, "errors": list(err.errors)})
                if dsl_validator is not None:
                    dsl_trace.append({
                        "round": round_idx,
                        "actor": "a",
                        "errors": list(err.errors),
                        "grammar_sha256": dsl_validator.grammar_sha256,
                    })
                return {
                    "status": "INVALID_DSL",
                    "rounds": round_idx,
                    "offender": "a",
                    "errors": list(err.errors),
                    "transcript": transcript,
                    "dsl_trace": dsl_trace,
                    "final_message": None,
                }

            validation_a = strat_a.validate_message(
                env_a,
                raw=raw_a,
                original=env_a_raw,
                transcript=transcript,
                actor="a",
                agent_name=getattr(agent_a, "name", "agent_a"),
            )
            env_a_processed, post_meta_a = strat_a.postprocess(
                env_a,
                raw=raw_a,
                validation=validation_a,
                transcript=transcript,
                actor="a",
                agent_name=getattr(agent_a, "name", "agent_a"),
            )
            if isinstance(env_a_processed, Envelope):
                env_a = env_a_processed
                env_a_data = env_a.model_dump()
            else:
                env_a, env_a_data, parse_a = _checked(env_a_processed, dsl_validator)

            stop_a, reason_a = strat_a.should_stop(
                env_a,
                validation=validation_a,
                transcript=transcript,
                actor="a",
                agent_name=getattr(agent_a, "name", "agent_a"),
            )

            _enforce_schema(env_a, schema_validator, getattr(agent_a, "name", "agent_a"), round_idx)

            transcript.append(
                {
                    "r": round_idx,
                    "actor": "a",
                    "envelope": env_a.model_dump(),
                    "raw": raw_a,
                    "strategy": {
                        "name": strat_a.name,
                        "max_rounds": strat_a.max_rounds,
                        "decoding": strat_a.decoding,
                        "metadata": strat_a.metadata,
                        "hooks": {
                            "prepare": prep_a,
                            "validation": validation_a,
                            "postprocess": post_meta_a,
                            "should_stop": {"stop": stop_a, "reason": reason_a},
                        },
                    },
                }
            )
            if parse_a is not None:
                trace_entry = parse_a.to_trace_entry(round_idx, "a")
                dsl_trace.append(trace_entry)
                last_parse["a"] = parse_a
            else:
                last_parse["a"] = None

            last_env["a"] = env_a
            solved_text["a"] = _final_return_value(env_a)

            parsed_intent = _parse_intent("A", env_a)
            if parsed_intent:
                intent_counts["a"][parsed_intent.intent] += 1

        # Agent B step
        prep_b = strat_b.prepare_prompt(
            task,
            transcript,
            actor="b",
            agent_name=getattr(agent_b, "name", "agent_b"),
        )
        env_b_raw, raw_b = _call_step(agent_b, task, transcript, prep_b)

        if text_mode:
            env_b = _prepare_text_turn(env_b_raw, validator)
            transcript.append({
                "r": round_idx,
                "actor": "b",
                "envelope": env_b,
                "raw": raw_b,
                "strategy": {
                    "name": strat_b.name,
                    "max_rounds": strat_b.max_rounds,
                    "decoding": strat_b.decoding,
                    "metadata": strat_b.metadata,
                    "hooks": {"prepare": prep_b},
                },
            })
            final_b = env_b.get("final_solution", {}).get("canonical_text") if isinstance(env_b, dict) else None
            solved_text["b"] = (final_b or "").strip() or None
        else:
            try:
                env_b, env_b_data, parse_b = _checked(env_b_raw, dsl_validator)
            except DSLValidationError as err:
                envelope = err.envelope if isinstance(err.envelope, dict) else {}
                transcript.append({"r": round_idx, "actor": "b", "envelope": envelope, "errors": list(err.errors)})
                if dsl_validator is not None:
                    dsl_trace.append({
                        "round": round_idx,
                        "actor": "b",
                        "errors": list(err.errors),
                        "grammar_sha256": dsl_validator.grammar_sha256,
                    })
                return {
                    "status": "INVALID_DSL",
                    "rounds": round_idx,
                    "offender": "b",
                    "errors": list(err.errors),
                    "transcript": transcript,
                    "dsl_trace": dsl_trace,
                    "final_message": None,
                }

            validation_b = strat_b.validate_message(
                env_b,
                raw=raw_b,
                original=env_b_raw,
                transcript=transcript,
                actor="b",
                agent_name=getattr(agent_b, "name", "agent_b"),
            )
            env_b_processed, post_meta_b = strat_b.postprocess(
                env_b,
                raw=raw_b,
                validation=validation_b,
                transcript=transcript,
                actor="b",
                agent_name=getattr(agent_b, "name", "agent_b"),
            )
            if isinstance(env_b_processed, Envelope):
                env_b = env_b_processed
                env_b_data = env_b.model_dump()
            else:
                env_b, env_b_data, parse_b = _checked(env_b_processed, dsl_validator)

            stop_b, reason_b = strat_b.should_stop(
                env_b,
                validation=validation_b,
                transcript=transcript,
                actor="b",
                agent_name=getattr(agent_b, "name", "agent_b"),
            )

            _enforce_schema(env_b, schema_validator, getattr(agent_b, "name", "agent_b"), round_idx)

            transcript.append(
                {
                    "r": round_idx,
                    "actor": "b",
                    "envelope": env_b.model_dump(),
                    "raw": raw_b,
                    "strategy": {
                        "name": strat_b.name,
                        "max_rounds": strat_b.max_rounds,
                        "decoding": strat_b.decoding,
                        "metadata": strat_b.metadata,
                        "hooks": {
                            "prepare": prep_b,
                            "validation": validation_b,
                            "postprocess": post_meta_b,
                            "should_stop": {"stop": stop_b, "reason": reason_b},
                        },
                    },
                }
            )
            if parse_b is not None:
                trace_entry = parse_b.to_trace_entry(round_idx, "b")
                dsl_trace.append(trace_entry)
                last_parse["b"] = parse_b
            else:
                last_parse["b"] = None

            last_env["b"] = env_b
            solved_text["b"] = _final_return_value(env_b)

            parsed_intent = _parse_intent("B", env_b)
            if parsed_intent:
                intent_counts["b"][parsed_intent.intent] += 1

        # Consensus checks
        if text_mode:
            text_a = solved_text["a"] or ""
            text_b = solved_text["b"] or ""
            if text_a and text_b:
                ca, ha = _canon_and_hash(text_a, kind)
                cb, hb = _canon_and_hash(text_b, kind)
                if ca == cb and ca:
                    final_message = {
                        "actor": "b",
                        "envelope": transcript[-1]["envelope"],
                    }
                    return {
                        "status": "CONSENSUS",
                        "rounds": round_idx,
                        "canonical_text": ca,
                        "sha256": ha,
                        "transcript": transcript,
                        "dsl_trace": dsl_trace,
                        "analytics": _intent_summary(intent_counts),
                        "final_message": final_message,
                    }
        else:
            env_a_latest = last_env["a"]
            env_b_latest = last_env["b"]
            if env_a_latest and env_b_latest:
                canon = _handshake_accept(env_b_latest, env_a_latest, kind)
                if canon:
                    final_message = {
                        "actor": "a",
                        "envelope": env_a_latest.model_dump(),
                    }
                    if last_parse["a"] is not None:
                        final_message["dsl"] = last_parse["a"].to_trace_entry(round_idx, "a")
                    ca, ha = _canon_and_hash(canon, kind)
                    return {
                        "status": "CONSENSUS",
                        "rounds": round_idx,
                        "canonical_text": ca,
                        "sha256": ha,
                        "transcript": transcript,
                        "dsl_trace": dsl_trace,
                        "analytics": _intent_summary(intent_counts),
                        "final_message": final_message,
                    }
                canon = _handshake_accept(env_a_latest, env_b_latest, kind)
                if canon:
                    final_message = {
                        "actor": "b",
                        "envelope": env_b_latest.model_dump(),
                    }
                    if last_parse["b"] is not None:
                        final_message["dsl"] = last_parse["b"].to_trace_entry(round_idx, "b")
                    cb, hb = _canon_and_hash(canon, kind)
                    return {
                        "status": "CONSENSUS",
                        "rounds": round_idx,
                        "canonical_text": cb,
                        "sha256": hb,
                        "transcript": transcript,
                        "dsl_trace": dsl_trace,
                        "analytics": _intent_summary(intent_counts),
                        "final_message": final_message,
                    }

            final_a = solved_text["a"]
            final_b = solved_text["b"]
            if final_a and final_b:
                ca, ha = _canon_and_hash(final_a, kind)
                cb, hb = _canon_and_hash(final_b, kind)
                if ca == cb and ca:
                    final_message = {
                        "actor": "b",
                        "envelope": env_b_latest.model_dump() if env_b_latest else {},
                    }
                    if last_parse["b"] is not None:
                        final_message["dsl"] = last_parse["b"].to_trace_entry(round_idx, "b")
                    return {
                        "status": "CONSENSUS",
                        "rounds": round_idx,
                        "canonical_text": ca,
                        "sha256": ha,
                        "transcript": transcript,
                        "dsl_trace": dsl_trace,
                        "analytics": _intent_summary(intent_counts),
                        "final_message": final_message,
                    }

    return {
        "status": "NO_CONSENSUS",
        "rounds": eff_max_rounds,
        "transcript": transcript,
        "dsl_trace": dsl_trace,
        "analytics": _intent_summary(intent_counts),
        "final_message": final_message,
    }


__all__ = ["run_controller"]
