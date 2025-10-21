from __future__ import annotations

import inspect
from collections import Counter
from dataclasses import asdict
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

from jsonschema import Draft7Validator
from pydantic import ValidationError

from .canonicalize import canonicalize_for_hash
from .control_trailer import normalise_canonical_text
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


def _register_control_error(stats: Dict[str, Any], code: str) -> None:
    if not isinstance(code, str) or not code:
        return
    stats.setdefault("error_log", []).append(code)
    if code == "missing_trailer":
        stats["trailer_missing_ct"] = stats.get("trailer_missing_ct", 0) + 1
    else:
        stats["invalid_trailer_ct"] = stats.get("invalid_trailer_ct", 0) + 1
    counts = stats.setdefault("error_counts", {})
    counts[code] = counts.get(code, 0) + 1
    if stats.get("first_error") is None:
        stats["first_error"] = code


def _update_control_stats(stats: Dict[str, Any], env: Envelope, round_idx: int) -> None:
    content = env.content or {}
    if not isinstance(content, dict):
        return
    control = content.get("control")
    if not isinstance(control, dict):
        stats["legacy_turns"] = stats.get("legacy_turns", 0) + 1
        return

    telemetry = control.get("telemetry")
    if isinstance(telemetry, dict):
        first_error = telemetry.get("first_error") or control.get("first_error")
        retry_count = telemetry.get("retry_count")
        body_len = telemetry.get("body_len")
        trailer_len = telemetry.get("trailer_len")
        stopped_on_ctrl = telemetry.get("stopped_on_ctrl")
        stop_reason = telemetry.get("stopped_on") or telemetry.get("stop_reason")
        tokens_reserved = telemetry.get("tokens_reserved") or telemetry.get("reserved_tokens")
        tokens_used_body = telemetry.get("tokens_used_body")
        tokens_used_trailer = telemetry.get("tokens_used_trailer")
        tokens_overflow = telemetry.get("tokens_overflow")
        tokens_body_budget = telemetry.get("tokens_body_budget") or telemetry.get("body_budget")
        tokens_body_overflow = telemetry.get("tokens_body_overflow")
        tokens_used_total = (
            telemetry.get("tokens_used_total")
            or telemetry.get("tokens_used")
            or telemetry.get("new_tokens")
        )
        has_tail = telemetry.get("has_tail")
        trailer_start = telemetry.get("trailer_start")
        trailer_end = telemetry.get("trailer_end")
        if telemetry.get("has_ctrl") and not telemetry.get("closed_ctrl"):
            _register_control_error(stats, "ERR_TRAILER_INCOMPLETE")
    else:
        first_error = control.get("first_error")
        retry_count = control.get("retry_count")
        body_len = control.get("body_len")
        trailer_len = control.get("trailer_len")
        stopped_on_ctrl = control.get("stopped_on_ctrl")
        stop_reason = control.get("stop_reason")
        tokens_reserved = control.get("tokens_reserved")
        tokens_used_body = control.get("tokens_used_body")
        tokens_used_trailer = control.get("tokens_used_trailer")
        tokens_overflow = control.get("tokens_overflow")
        tokens_body_budget = control.get("tokens_body_budget")
        tokens_body_overflow = control.get("tokens_body_overflow")
        tokens_used_total = control.get("tokens_used_total")
        has_tail = control.get("has_tail")
        trailer_start = control.get("trailer_start")
        trailer_end = control.get("trailer_end")

    errors = control.get("errors")

    if isinstance(retry_count, int):
        stats["retry_count"] = stats.get("retry_count", 0) + max(retry_count, 0)

    if stats.get("first_error") is None and isinstance(first_error, str) and first_error:
        stats["first_error"] = first_error

    if isinstance(errors, list):
        for entry in errors:
            if isinstance(entry, str):
                _register_control_error(stats, entry)

    if isinstance(body_len, int):
        stats["body_len_total"] = stats.get("body_len_total", 0) + max(body_len, 0)
        stats["body_len_count"] = stats.get("body_len_count", 0) + 1

    if isinstance(trailer_len, int):
        stats["trailer_len_total"] = stats.get("trailer_len_total", 0) + max(trailer_len, 0)
        stats["trailer_len_count"] = stats.get("trailer_len_count", 0) + 1

    if isinstance(tokens_reserved, int):
        stats["tokens_reserved_total"] = stats.get("tokens_reserved_total", 0) + max(tokens_reserved, 0)
        stats["tokens_reserved_count"] = stats.get("tokens_reserved_count", 0) + 1

    if isinstance(tokens_used_body, int):
        stats["tokens_used_body_total"] = stats.get("tokens_used_body_total", 0) + max(tokens_used_body, 0)

    if isinstance(tokens_used_trailer, int):
        stats["tokens_used_trailer_total"] = stats.get("tokens_used_trailer_total", 0) + max(tokens_used_trailer, 0)

    if isinstance(tokens_used_total, int):
        stats["tokens_used_total"] = stats.get("tokens_used_total", 0) + max(tokens_used_total, 0)

    if isinstance(tokens_overflow, int) and tokens_overflow > 0:
        stats["overflow_turns"] = stats.get("overflow_turns", 0) + 1
        stats["max_overflow"] = max(stats.get("max_overflow", 0), tokens_overflow)
        stats["needs_higher_reserve"] = True

    if isinstance(tokens_body_overflow, int) and tokens_body_overflow > 0:
        stats["body_overflow_turns"] = stats.get("body_overflow_turns", 0) + 1

    if isinstance(stopped_on_ctrl, bool) and stopped_on_ctrl:
        stats["stopped_on_ctrl_ct"] = stats.get("stopped_on_ctrl_ct", 0) + 1

    if isinstance(stop_reason, str):
        normalized = stop_reason.lower()
        if normalized in {"ctrl", "suffix"}:
            key = "suffix"
            if not isinstance(stopped_on_ctrl, bool) or not stopped_on_ctrl:
                stats["stopped_on_ctrl_ct"] = stats.get("stopped_on_ctrl_ct", 0) + 1
        elif normalized == "eos":
            key = "eos"
        elif normalized in {"length", "max_new_tokens"}:
            key = "max_new_tokens"
            stats["needs_higher_reserve"] = True
        else:
            key = None

        if key:
            stop_counts = stats.setdefault("stop_reasons", {})
            stop_counts[key] = stop_counts.get(key, 0) + 1

    if env.final_solution and env.status in {"PROPOSED", "READY_TO_SOLVE"}:
        stats.setdefault("first_valid_round", round_idx)

    if has_tail:
        _register_control_error(stats, "not_at_end")

    if isinstance(trailer_start, int) and isinstance(trailer_end, int):
        offsets = stats.setdefault("trailer_offsets", [])
        offsets.append({"round": round_idx, "start": trailer_start, "end": trailer_end})


def _control_summary(stats: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "trailer_missing_ct": stats.get("trailer_missing_ct", 0),
        "invalid_trailer_ct": stats.get("invalid_trailer_ct", 0),
        "retry_count": stats.get("retry_count", 0),
    }
    if stats.get("first_error"):
        summary["first_error"] = stats["first_error"]
    if stats.get("error_log"):
        summary["error_log"] = list(stats["error_log"])
    if stats.get("stopped_on_ctrl_ct"):
        summary["stopped_on_ctrl_ct"] = stats["stopped_on_ctrl_ct"]
        summary.setdefault("stopped_on_ctrl", stats["stopped_on_ctrl_ct"])
    if stats.get("handshake_error_ct"):
        summary["handshake_error_ct"] = stats["handshake_error_ct"]
    if stats.get("legacy_turns"):
        summary["legacy_turns"] = stats["legacy_turns"]
    if stats.get("overflow_turns"):
        summary["overflow_turns"] = stats["overflow_turns"]
    if stats.get("max_overflow"):
        summary["max_overflow"] = stats["max_overflow"]
    if stats.get("body_overflow_turns"):
        summary["body_overflow_turns"] = stats["body_overflow_turns"]
    if stats.get("needs_higher_reserve"):
        summary["needs_higher_reserve"] = bool(stats["needs_higher_reserve"])
    if stats.get("solved_round"):
        summary["solved_round"] = stats["solved_round"]
    if stats.get("proposer"):
        summary["proposer"] = stats["proposer"]
    if stats.get("acceptor"):
        summary["acceptor"] = stats["acceptor"]
    if stats.get("final_canonical"):
        summary["final_canonical"] = stats["final_canonical"]
    body_count = stats.get("body_len_count", 0)
    if body_count:
        summary["avg_body_len"] = stats.get("body_len_total", 0) / body_count
    trailer_count = stats.get("trailer_len_count", 0)
    if trailer_count:
        summary["avg_trailer_len"] = stats.get("trailer_len_total", 0) / trailer_count
    tokens_reserved_total = stats.get("tokens_reserved_total")
    tokens_reserved_count = stats.get("tokens_reserved_count")
    if tokens_reserved_total and tokens_reserved_count:
        summary["avg_tokens_reserved"] = tokens_reserved_total / max(tokens_reserved_count, 1)
    if stats.get("tokens_used_trailer_total"):
        summary["tokens_used_trailer_total"] = stats["tokens_used_trailer_total"]
    if stats.get("tokens_used_body_total"):
        summary["tokens_used_body_total"] = stats["tokens_used_body_total"]
    if stats.get("tokens_used_total"):
        summary["tokens_used_total"] = stats["tokens_used_total"]
    if stats.get("first_valid_round"):
        summary["first_valid_round"] = stats["first_valid_round"]
    if stats.get("first_proposal_round"):
        summary["first_proposal_round"] = stats["first_proposal_round"]
    stop_reasons = stats.get("stop_reasons")
    if isinstance(stop_reasons, dict):
        for reason, count in stop_reasons.items():
            summary[f"stopped_on_{reason}"] = count
    if stats.get("error_counts"):
        summary["error_counts"] = dict(stats["error_counts"])
    if stats.get("trailer_offsets"):
        summary["trailer_offsets"] = list(stats["trailer_offsets"])
    return summary


class HandshakeTracker:
    """Track proposal/acceptance transitions for SOLVED handshakes."""

    def __init__(self) -> None:
        self.pending_actor: Optional[str] = None
        self.pending_canonical: Optional[str] = None
        self.pending_round: Optional[int] = None

    def observe(self, actor: str, env: Envelope, round_idx: int) -> Optional[Dict[str, Any]]:
        status = (env.status or "").strip().upper()
        tag = (env.tag or "").strip().upper()
        final = env.final_solution
        canonical: Optional[str] = None
        if final and isinstance(final.canonical_text, str):
            canonical = normalise_canonical_text(final.canonical_text)
            if canonical:
                canonical = canonical.strip()

        if canonical and status not in {"PROPOSED", "READY_TO_SOLVE", "SOLVED"} and tag != "[SOLVED]":
            return {
                "kind": "error",
                "actor": actor,
                "error": "illegal_transition",
                "round": round_idx,
            }

        if status in {"PROPOSED", "READY_TO_SOLVE"}:
            if canonical:
                self.pending_actor = actor
                self.pending_canonical = canonical
                self.pending_round = round_idx
                return {
                    "kind": "proposal",
                    "actor": actor,
                    "canonical": canonical,
                    "round": round_idx,
                }
            return {
                "kind": "error",
                "actor": actor,
                "error": "missing_canonical",
                "round": round_idx,
            }

        if tag == "[SOLVED]" and status == "SOLVED":
            if not canonical:
                return {
                    "kind": "error",
                    "actor": actor,
                    "error": "missing_canonical",
                    "round": round_idx,
                }
            if not self.pending_actor or not self.pending_canonical:
                return {
                    "kind": "error",
                    "actor": actor,
                    "error": "illegal_transition",
                    "round": round_idx,
                }
            if self.pending_actor == actor:
                return {
                    "kind": "error",
                    "actor": actor,
                    "error": "illegal_transition",
                    "round": round_idx,
                }
            if normalise_canonical_text(canonical) == normalise_canonical_text(self.pending_canonical):
                accepted = {
                    "kind": "accepted",
                    "actor": actor,
                    "canonical": normalise_canonical_text(canonical),
                    "proposer": self.pending_actor,
                    "round": round_idx,
                }
                self.pending_actor = None
                self.pending_canonical = None
                self.pending_round = None
                return accepted
            return {
                "kind": "error",
                "actor": actor,
                "error": "illegal_transition",
                "round": round_idx,
            }

        return None


def _handle_handshake_event(stats: Dict[str, Any], event: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not event:
        return None
    kind = event.get("kind")
    if kind == "proposal":
        round_idx = event.get("round")
        if stats.get("first_proposal_round") is None and isinstance(round_idx, int):
            stats["first_proposal_round"] = round_idx
        stats["proposer"] = event.get("actor")
        stats["pending_canonical"] = event.get("canonical")
        return None
    if kind == "error":
        code = event.get("error")
        if isinstance(code, str):
            _register_control_error(stats, code)
            stats["handshake_error_ct"] = stats.get("handshake_error_ct", 0) + 1
            log = stats.setdefault("handshake_error_log", [])
            log.append({k: event.get(k) for k in ("round", "actor", "error")})
        return None
    if kind == "accepted":
        round_idx = event.get("round")
        if isinstance(round_idx, int):
            stats["solved_round"] = round_idx
        stats["acceptor"] = event.get("actor")
        if event.get("proposer"):
            stats["proposer"] = event.get("proposer")
        if event.get("canonical"):
            stats["final_canonical"] = event.get("canonical")
        return event
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
    control_stats: Dict[str, Any] = {
        "trailer_missing_ct": 0,
        "invalid_trailer_ct": 0,
        "retry_count": 0,
        "first_error": None,
        "error_log": [],
    }
    handshake = HandshakeTracker()

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
            _update_control_stats(control_stats, env_a, round_idx)

            accepted_a = _handle_handshake_event(control_stats, handshake.observe("a", env_a, round_idx))
            if accepted_a:
                canonical = str(accepted_a.get("canonical") or "").strip()
                ca, ha = _canon_and_hash(canonical, kind)
                final_message = {
                    "actor": "a",
                    "envelope": env_a.model_dump(),
                    "canonical_text": ca,
                }
                if last_parse["a"] is not None:
                    final_message["dsl"] = last_parse["a"].to_trace_entry(round_idx, "a")
                analytics = _intent_summary(intent_counts)
                analytics["control"] = _control_summary(control_stats)
                return {
                    "status": "CONSENSUS",
                    "rounds": round_idx,
                    "canonical_text": ca,
                    "sha256": ha,
                    "transcript": transcript,
                    "dsl_trace": dsl_trace,
                    "analytics": analytics,
                    "final_message": final_message,
                }

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
            _update_control_stats(control_stats, env_b, round_idx)

            accepted_b = _handle_handshake_event(control_stats, handshake.observe("b", env_b, round_idx))
            if accepted_b:
                canonical = str(accepted_b.get("canonical") or "").strip()
                cb, hb = _canon_and_hash(canonical, kind)
                final_message = {
                    "actor": "b",
                    "envelope": env_b.model_dump(),
                    "canonical_text": cb,
                }
                if last_parse["b"] is not None:
                    final_message["dsl"] = last_parse["b"].to_trace_entry(round_idx, "b")
                analytics = _intent_summary(intent_counts)
                analytics["control"] = _control_summary(control_stats)
                return {
                    "status": "CONSENSUS",
                    "rounds": round_idx,
                    "canonical_text": cb,
                    "sha256": hb,
                    "transcript": transcript,
                    "dsl_trace": dsl_trace,
                    "analytics": analytics,
                    "final_message": final_message,
                }

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
                        "canonical_text": ca,
                    }
                    analytics = _intent_summary(intent_counts)
                    analytics["control"] = _control_summary(control_stats)
                    return {
                        "status": "CONSENSUS",
                        "rounds": round_idx,
                        "canonical_text": ca,
                        "sha256": ha,
                        "transcript": transcript,
                        "dsl_trace": dsl_trace,
                        "analytics": analytics,
                        "final_message": final_message,
                    }
        else:
            env_a_latest = last_env["a"]
            env_b_latest = last_env["b"]
            final_a = solved_text["a"]
            final_b = solved_text["b"]
            if final_a and final_b:
                ca, ha = _canon_and_hash(final_a, kind)
                cb, hb = _canon_and_hash(final_b, kind)
                if ca == cb and ca:
                    final_message = {
                        "actor": "b",
                        "envelope": env_b_latest.model_dump() if env_b_latest else {},
                        "canonical_text": ca,
                    }
                    if last_parse["b"] is not None:
                        final_message["dsl"] = last_parse["b"].to_trace_entry(round_idx, "b")
                    analytics = _intent_summary(intent_counts)
                    analytics["control"] = _control_summary(control_stats)
                    return {
                        "status": "CONSENSUS",
                        "rounds": round_idx,
                        "canonical_text": ca,
                        "sha256": ha,
                        "transcript": transcript,
                        "dsl_trace": dsl_trace,
                        "analytics": analytics,
                        "final_message": final_message,
                    }

    analytics = _intent_summary(intent_counts)
    analytics["control"] = _control_summary(control_stats)
    return {
        "status": "NO_CONSENSUS",
        "rounds": eff_max_rounds,
        "transcript": transcript,
        "dsl_trace": dsl_trace,
        "analytics": analytics,
        "final_message": final_message,
    }


__all__ = ["run_controller"]
