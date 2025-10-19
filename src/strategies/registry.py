from __future__ import annotations

"""Central registry describing available communication strategies."""

import copy
from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Tuple

from ..pseudocode import PSEUDOCODE_INSERT
from ..utils import parse_acl_message

PreRoundHook = Callable[[MutableMapping[str, Any]], None]
EnvelopeValidator = Callable[[Mapping[str, Any]], Tuple[bool, Optional[str]]]
PromptDecorator = Callable[[str, Mapping[str, Any]], str]
ControllerBehavior = Callable[[MutableMapping[str, Any]], None]


@dataclass(frozen=True)
class AgentProfile:
    """Lightweight configuration for agent sampling behaviour."""

    greedy: bool = True
    k_samples: int = 1
    max_new_tokens: int = 256

    def clone(self) -> "AgentProfile":
        return replace(self)


@dataclass
class Strategy:
    id: str
    name: str
    description: str = ""
    json_only: bool = True
    allow_cot: bool = False
    max_rounds: int = 8
    decoding: Dict[str, Any] = field(default_factory=dict)
    consensus_mode: str = "review_handshake"
    pre_round_hooks: Tuple[PreRoundHook, ...] = field(default_factory=tuple)
    envelope_validators: Tuple[EnvelopeValidator, ...] = field(default_factory=tuple)
    prompt_decorators: Tuple[PromptDecorator, ...] = field(default_factory=tuple)
    controller_behaviors: Tuple[ControllerBehavior, ...] = field(default_factory=tuple)
    agent_profile: AgentProfile = field(default_factory=AgentProfile)
    metadata: Dict[str, Any] = field(default_factory=dict)
    output_mode: str = "json"

    def apply_pre_round_hooks(self, controller_state: MutableMapping[str, Any]) -> None:
        for hook in self.pre_round_hooks:
            hook(controller_state)

    def apply_controller_behaviors(self, controller_state: MutableMapping[str, Any]) -> None:
        for behavior in self.controller_behaviors:
            behavior(controller_state)

    def validate_envelope(self, envelope: Mapping[str, Any]) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        for validator in self.envelope_validators:
            ok, message = validator(envelope)
            if not ok:
                errors.append(message or "Envelope validation failed.")
        return len(errors) == 0, errors

    def decorate_prompts(self, prompt: str, context: Optional[Mapping[str, Any]] = None) -> str:
        decorated = prompt
        ctx = context or {}
        for decorator in self.prompt_decorators:
            decorated = decorator(decorated, ctx)
        return decorated

    @property
    def agent_defaults(self) -> AgentProfile:
        return self.agent_profile.clone()


@dataclass(frozen=True)
class StrategyDefinition:
    id: str
    name: str
    description: str = ""
    json_only: bool = True
    allow_cot: bool = False
    max_rounds: int = 8
    decoding: Mapping[str, Any] = field(
        default_factory=lambda: {
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": 256,
        }
    )
    consensus_mode: str = "review_handshake"
    pre_round_hooks: Tuple[PreRoundHook, ...] = field(default_factory=tuple)
    envelope_validators: Tuple[EnvelopeValidator, ...] = field(default_factory=tuple)
    prompt_decorators: Tuple[PromptDecorator, ...] = field(default_factory=tuple)
    controller_behaviors: Tuple[ControllerBehavior, ...] = field(default_factory=tuple)
    agent_profile: AgentProfile = field(default_factory=AgentProfile)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    output_mode: str = "json"

    def instantiate(self, **overrides: Any) -> Strategy:
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data.update(overrides)

        decoding = data.get("decoding") or {}
        data["decoding"] = copy.deepcopy(dict(decoding))

        metadata = data.get("metadata") or {}
        data["metadata"] = copy.deepcopy(dict(metadata))

        agent_profile = data.get("agent_profile")
        if isinstance(agent_profile, AgentProfile):
            data["agent_profile"] = agent_profile.clone()

        def _tupled(value: Any) -> Tuple[Any, ...]:
            if not value:
                return tuple()
            return tuple(value)

        data["pre_round_hooks"] = _tupled(data.get("pre_round_hooks"))
        data["envelope_validators"] = _tupled(data.get("envelope_validators"))
        data["prompt_decorators"] = _tupled(data.get("prompt_decorators"))
        data["controller_behaviors"] = _tupled(data.get("controller_behaviors"))

        return Strategy(**data)


# ---------------------------------------------------------------------------
# Built-in behaviours and validators


def _set_controller_meta(key: str, value: Any) -> ControllerBehavior:
    def _apply(state: MutableMapping[str, Any]) -> None:
        state.setdefault("meta", {})[key] = value

    return _apply


def _ensure_field(name: str) -> EnvelopeValidator:
    def _validator(envelope: Mapping[str, Any]) -> Tuple[bool, Optional[str]]:
        value = envelope.get(name)
        if isinstance(value, str) and value.strip():
            return True, None
        return False, f"Envelope missing '{name}' field."

    return _validator


def _ensure_final_solution(envelope: Mapping[str, Any]) -> Tuple[bool, Optional[str]]:
    status = str(envelope.get("status") or "").upper()
    final = envelope.get("final_solution")
    if status == "SOLVED":
        if not isinstance(final, Mapping):
            return False, "final_solution must be present when status is SOLVED."
        text = final.get("canonical_text") if isinstance(final, Mapping) else None
        if not isinstance(text, str) or not text.strip():
            return False, "final_solution.canonical_text is required for SOLVED messages."
    return True, None


def _ensure_acl_valid(envelope: Mapping[str, Any]) -> Tuple[bool, Optional[str]]:
    content = envelope.get("content") if isinstance(envelope.get("content"), Mapping) else None
    acl = content.get("acl") if isinstance(content, Mapping) else None
    if not isinstance(acl, str):
        return False, "content.acl must be a string."
    try:
        parse_acl_message(acl)
    except Exception as exc:  # pragma: no cover - error message captured by controller
        return False, str(exc)
    return True, None


def _decorate_append(text: str) -> PromptDecorator:
    def _decorator(prompt: str, _: Mapping[str, Any]) -> str:
        if text in prompt:
            return prompt
        return prompt.rstrip() + "\n\n" + text

    return _decorator


def _decorate_json_hint(prompt: str, _: Mapping[str, Any]) -> str:
    suffix = "Remember to answer with a single valid JSON object."
    if suffix in prompt:
        return prompt
    return prompt.rstrip() + "\n\n" + suffix


# ---------------------------------------------------------------------------
# Registry definitions

STRATEGY_REGISTRY: Dict[str, StrategyDefinition] = {}


def register_strategy(definition: StrategyDefinition) -> None:
    STRATEGY_REGISTRY[definition.id] = definition


# Natural language ---------------------------------------------------------

register_strategy(
    StrategyDefinition(
        id="natural_language",
        name="Natural Language",
        description="Concise natural language coordination with no JSON envelope.",
        json_only=False,
        allow_cot=True,
        max_rounds=6,
        decoding={
            "do_sample": False,
            "temperature": 0.2,
            "max_new_tokens": 256,
        },
        prompt_decorators=(
            _decorate_append(
                "Use clear, concise natural language. Avoid JSON unless explicitly requested."
            ),
        ),
        controller_behaviors=(
            _set_controller_meta("strategy_id", "natural_language"),
            _set_controller_meta("json_only", False),
            _set_controller_meta("output_mode", "text"),
        ),
        agent_profile=AgentProfile(greedy=True, k_samples=1, max_new_tokens=256),
        metadata={
            "style": "natural_language",
            "output_mode": "text",
        },
        output_mode="text",
    )
)


# JSON + schema ------------------------------------------------------------

register_strategy(
    StrategyDefinition(
        id="json_schema",
        name="JSON with Schema",
        description="Strict JSON envelopes validated against the shared schema.",
        json_only=True,
        allow_cot=False,
        max_rounds=8,
        decoding={
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": 256,
        },
        envelope_validators=(
            _ensure_field("status"),
            _ensure_field("tag"),
            _ensure_final_solution,
        ),
        prompt_decorators=(
            _decorate_json_hint,
        ),
        controller_behaviors=(
            _set_controller_meta("strategy_id", "json_schema"),
            _set_controller_meta("json_only", True),
            _set_controller_meta("output_mode", "json"),
        ),
        agent_profile=AgentProfile(greedy=True, k_samples=1, max_new_tokens=256),
        metadata={
            "schema": "schemas/envelope.schema.json",
            "output_mode": "json",
        },
        output_mode="json",
    )
)


# Pseudocode ---------------------------------------------------------------

register_strategy(
    StrategyDefinition(
        id="pseudocode",
        name="Pseudocode",
        description="JSON envelopes whose final_solution contains strict pseudocode.",
        json_only=True,
        allow_cot=True,
        max_rounds=10,
        decoding={
            "do_sample": False,
            "temperature": 0.1,
            "max_new_tokens": 384,
        },
        envelope_validators=(
            _ensure_field("status"),
            _ensure_field("tag"),
            _ensure_final_solution,
        ),
        prompt_decorators=(
            _decorate_append(PSEUDOCODE_INSERT),
            _decorate_json_hint,
        ),
        controller_behaviors=(
            _set_controller_meta("strategy_id", "pseudocode"),
            _set_controller_meta("json_only", True),
            _set_controller_meta("output_mode", "json"),
            _set_controller_meta("requires_pseudocode", True),
        ),
        agent_profile=AgentProfile(greedy=True, k_samples=1, max_new_tokens=384),
        metadata={
            "requires_pseudocode": True,
            "output_mode": "json",
        },
        output_mode="json",
    )
)


# Symbolic ACL -------------------------------------------------------------

register_strategy(
    StrategyDefinition(
        id="symbolic_acl",
        name="Symbolic Agent Language",
        description="JSON envelopes with ACL-performative coordination messages.",
        json_only=True,
        allow_cot=False,
        max_rounds=8,
        decoding={
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": 256,
        },
        envelope_validators=(
            _ensure_field("status"),
            _ensure_field("tag"),
            _ensure_acl_valid,
        ),
        prompt_decorators=(
            _decorate_json_hint,
            _decorate_append(
                "Structure coordination in the form 'INTENT: message => NEXT'."
            ),
        ),
        controller_behaviors=(
            _set_controller_meta("strategy_id", "symbolic_acl"),
            _set_controller_meta("json_only", True),
            _set_controller_meta("output_mode", "json"),
            _set_controller_meta("requires_acl", True),
        ),
        agent_profile=AgentProfile(greedy=True, k_samples=1, max_new_tokens=256),
        metadata={
            "requires_acl": True,
            "output_mode": "json",
        },
        output_mode="json",
    )
)


# Emergent DSL -------------------------------------------------------------

register_strategy(
    StrategyDefinition(
        id="emergent_dsl",
        name="Emergent DSL",
        description="Text-only communication using the shared DSL grammar.",
        json_only=False,
        allow_cot=True,
        max_rounds=12,
        decoding={
            "do_sample": True,
            "temperature": 0.4,
            "top_p": 0.9,
            "max_new_tokens": 256,
        },
        prompt_decorators=(
            _decorate_append(
                (
                    "Use the grammar 'INTENT: content => NEXT_ACTION'. Allowed intents: "
                    "DEFINE, PLAN, EXECUTE, REVISE, ASK, CONFIRM, SOLVED."
                )
            ),
        ),
        controller_behaviors=(
            _set_controller_meta("strategy_id", "emergent_dsl"),
            _set_controller_meta("json_only", False),
            _set_controller_meta("output_mode", "text"),
            _set_controller_meta("requires_dsl", True),
        ),
        agent_profile=AgentProfile(greedy=False, k_samples=1, max_new_tokens=256),
        metadata={
            "requires_dsl": True,
            "output_mode": "text",
        },
        output_mode="text",
    )
)


__all__ = [
    "AgentProfile",
    "Strategy",
    "StrategyDefinition",
    "STRATEGY_REGISTRY",
    "register_strategy",
]
