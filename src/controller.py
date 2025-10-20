from __future__ import annotations

"""Conversation controller orchestrating two agents under a strategy."""

from collections import Counter
from typing import Any, Dict, List, MutableMapping, Optional, Tuple

from .json_enforcer import validate_envelope
from .pseudocode import PseudocodeValidationError, validate_and_normalise_pseudocode
from .schemas import Envelope
from .strategies import Strategy, build_strategy
from .utils import (
    ACLParseError,
    normalize_text,
    parse_acl_message,
    parse_dsl_message,
    sha256_hex,
)


def _strategy_for(agent) -> Strategy:
    strategy = getattr(agent, "strategy", None)
    if isinstance(strategy, Strategy):
        return strategy
    strategy_id = getattr(agent, "strategy_id", None) or "json_schema"
    return build_strategy(strategy_id)


def _prepare_context(strategy: Strategy, actor_label: str, round_index: int) -> MutableMapping[str, Any]:
    ctx: MutableMapping[str, Any] = {"round": round_index, "actor": actor_label, "meta": {}}
    strategy.apply_pre_round_hooks(ctx)
    strategy.apply_controller_behaviors(ctx)
    return ctx


def _record_acl_intent(analytics: Dict[str, Counter], actor_key: str, acl_text: str) -> Dict[str, Any]:
    parsed = parse_acl_message(acl_text)
    analytics.setdefault(actor_key, Counter())[parsed.intent] += 1
    return {"intent": parsed.intent, "content": parsed.content, "next_action": parsed.next_action}


def _process_json_message(
    strategy: Strategy,
    message: Dict[str, Any],
    *,
    actor_key: str,
    analytics: Dict[str, Counter],
    schema_validator,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ok, errors = strategy.validate_envelope(message)
    if not ok:
        raise ValueError(
            f"Strategy '{strategy.id}' envelope validation failed: {'; '.join(errors)}"
        )

    envelope = Envelope.model_validate(message)
    payload = envelope.model_dump(exclude_none=True)
    payload.setdefault("meta", {})["strategy_id"] = strategy.id

    if schema_validator is not None:
        valid, schema_errors = validate_envelope(payload, schema_validator)
        if not valid:
            raise ValueError(
                f"Schema validation failed for strategy '{strategy.id}': {'; '.join(schema_errors)}"
            )

    extras: Dict[str, Any] = {}

    if strategy.metadata.get("requires_acl"):
        acl_text = (payload.get("content") or {}).get("acl", "")
        if not isinstance(acl_text, str):
            raise ValueError("ACL content must be a string when the strategy requires ACL messages.")
        try:
            extras["acl"] = _record_acl_intent(analytics, actor_key, acl_text)
        except ACLParseError as exc:
            raise ValueError(f"Invalid ACL message from agent {actor_key}: {exc}") from exc

    if strategy.metadata.get("requires_pseudocode") and payload.get("final_solution"):
        canonical = payload["final_solution"].get("canonical_text", "")
        if canonical:
            try:
                normalised, return_value = validate_and_normalise_pseudocode(canonical)
            except PseudocodeValidationError as exc:
                raise ValueError(f"Invalid pseudocode from agent {actor_key}: {exc}") from exc
            payload.setdefault("final_solution", {})["canonical_text"] = normalised
            payload["final_solution"]["return_value"] = return_value

    return payload, extras


def _process_text_message(strategy: Strategy, message: Any) -> Tuple[str, Dict[str, Any]]:
    text = normalize_text(message)
    extras: Dict[str, Any] = {}
    if strategy.metadata.get("requires_dsl"):
        extras["dsl"] = parse_dsl_message(text)
    return text, extras


def run_controller(
    task: str,
    agent_a,
    agent_b,
    *,
    max_rounds: Optional[int] = None,
    schema_validator=None,
) -> Dict[str, Any]:
    strategy_a = _strategy_for(agent_a)
    strategy_b = _strategy_for(agent_b)
    total_rounds = max_rounds or max(strategy_a.max_rounds, strategy_b.max_rounds)

    transcript: List[Dict[str, Any]] = []
    analytics: Dict[str, Counter] = {"a": Counter(), "b": Counter()}
    final_json: Dict[str, str] = {}
    final_text: Dict[str, str] = {}
    final_dsl: Dict[str, str] = {}
    final_entries: Dict[str, Dict[str, Any]] = {}

    agents = [(agent_a, strategy_a, "a"), (agent_b, strategy_b, "b")]

    for round_index in range(1, total_rounds + 1):
        for agent, strategy, actor_key in agents:
            context = _prepare_context(strategy, actor_key, round_index)
            message, raw = agent.step(task, transcript)

            entry: Dict[str, Any] = {
                "round": round_index,
                "actor": getattr(agent, "name", actor_key),
                "actor_key": actor_key,
                "strategy": {
                    "id": strategy.id,
                    "name": strategy.name,
                    "metadata": strategy.metadata,
                },
                "raw": raw,
                "context": context,
            }

            if strategy.json_only:
                if not isinstance(message, dict):
                    raise TypeError(
                        f"Agent '{entry['actor']}' returned non-dict payload under JSON strategy."
                    )
                payload, extras = _process_json_message(
                    strategy,
                    message,
                    actor_key=actor_key,
                    analytics=analytics,
                    schema_validator=schema_validator,
                )
                entry["envelope"] = payload
                entry["strategy"].update(extras)
                final_solution = payload.get("final_solution") if isinstance(payload, dict) else None
                if isinstance(final_solution, dict) and final_solution.get("canonical_text"):
                    final_json[actor_key] = final_solution["canonical_text"]
                final_entries[actor_key] = entry
            else:
                text, extras = _process_text_message(strategy, message)
                entry["text"] = text
                entry["strategy"].update(extras)

                if strategy.metadata.get("requires_dsl"):
                    dsl_info = extras.get("dsl", {}) if isinstance(extras, dict) else {}
                    if (
                        isinstance(dsl_info, dict)
                        and dsl_info.get("intent") == "SOLVED"
                        and isinstance(dsl_info.get("content"), str)
                        and dsl_info["content"].strip()
                    ):
                        final_dsl[actor_key] = dsl_info["content"].strip()
                else:
                    if text:
                        final_text[actor_key] = text
                final_entries[actor_key] = entry

            transcript.append(entry)

    status = "NO_CONSENSUS"
    canonical_text: Optional[str] = None
    if final_json.get("a") and final_json.get("b") and final_json["a"] == final_json["b"]:
        canonical_text = final_json["a"]
        status = "CONSENSUS"
    elif final_dsl.get("a") and final_dsl.get("b") and final_dsl["a"] == final_dsl["b"]:
        canonical_text = final_dsl["a"]
        status = "CONSENSUS"
    elif final_text.get("a") and final_text.get("b") and final_text["a"] == final_text["b"]:
        canonical_text = final_text["a"]
        status = "CONSENSUS"

    result: Dict[str, Any] = {
        "status": status,
        "rounds": len(transcript),
        "transcript": transcript,
        "analytics": {actor: dict(counter) for actor, counter in analytics.items()},
        "final_messages": {actor: final_entries.get(actor) for actor in ("a", "b")},
    }

    if canonical_text:
        result["canonical_text"] = canonical_text
        result["sha256"] = sha256_hex(canonical_text)

    return result
