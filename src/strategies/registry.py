from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Tuple

PreRoundHook = Callable[[MutableMapping[str, Any]], None]
EnvelopeValidator = Callable[[Mapping[str, Any]], Tuple[bool, Optional[str]]]
PromptDecorator = Callable[[str, Mapping[str, Any]], str]
ControllerBehavior = Callable[[MutableMapping[str, Any]], None]


@dataclass(frozen=True)
class AgentProfile:
    """Lightweight configuration for sampling behaviour of interactive agents."""

    greedy: bool = True
    k_samples: int = 1
    max_new_tokens: int = 256

    def clone(self) -> "AgentProfile":
        """Return a shallow copy that can be mutated by callers without affecting the registry."""

        return replace(self)


@dataclass
class Strategy:
    """Runtime strategy object surfaced to controllers and agents."""

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

    def apply_pre_round_hooks(self, controller_state: MutableMapping[str, Any]) -> None:
        for hook in self.pre_round_hooks:
            hook(controller_state)

    def apply_controller_behaviors(self, controller_state: MutableMapping[str, Any]) -> None:
        for behavior in self.controller_behaviors:
            behavior(controller_state)

    def validate_envelope(self, envelope: Mapping[str, Any]) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        for validator in self.envelope_validators:
            valid, message = validator(envelope)
            if not valid:
                errors.append(message or "Envelope validation failed.")
        return len(errors) == 0, errors

    def decorate_prompts(self, prompt: str, context: Optional[Mapping[str, Any]] = None) -> str:
        ctx = context or {}
        decorated = prompt
        for decorator in self.prompt_decorators:
            decorated = decorator(decorated, ctx)
        return decorated

    @property
    def agent_defaults(self) -> AgentProfile:
        """Expose a copy of the agent profile defaults for consumer modules."""

        return self.agent_profile.clone()


@dataclass(frozen=True)
class StrategyDefinition:
    """Immutable template used to instantiate runtime strategy objects."""

    id: str
    name: str
    description: str = ""
    json_only: bool = True
    allow_cot: bool = False
    max_rounds: int = 8
    decoding: Mapping[str, Any] = field(default_factory=lambda: {
        "do_sample": False,
        "temperature": 0.0,
        "max_new_tokens": 256,
    })
    consensus_mode: str = "review_handshake"
    pre_round_hooks: Tuple[PreRoundHook, ...] = field(default_factory=tuple)
    envelope_validators: Tuple[EnvelopeValidator, ...] = field(default_factory=tuple)
    prompt_decorators: Tuple[PromptDecorator, ...] = field(default_factory=tuple)
    controller_behaviors: Tuple[ControllerBehavior, ...] = field(default_factory=tuple)
    agent_profile: AgentProfile = field(default_factory=AgentProfile)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def instantiate(self, **overrides: Any) -> Strategy:
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data.update(overrides)

        # Ensure mapping values are copied so mutations do not affect the registry.
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


# --- Built-in behaviours ----------------------------------------------------

def _ensure_status(envelope: Mapping[str, Any]) -> Tuple[bool, Optional[str]]:
    if isinstance(envelope.get("status"), str) and envelope["status"].strip():
        return True, None
    return False, "Envelope missing 'status' field."


def _ensure_tag(envelope: Mapping[str, Any]) -> Tuple[bool, Optional[str]]:
    tag = envelope.get("tag")
    if isinstance(tag, str) and tag.strip():
        return True, None
    return False, "Envelope missing 'tag' field."


def _decorate_with_json_hint(prompt: str, context: Mapping[str, Any]) -> str:
    if context.get("json_hint_applied"):
        return prompt
    suffix = "\n\nRemember to answer with a single valid JSON object."
    return prompt + suffix


def _decorate_with_guidance(message: str) -> PromptDecorator:
    def _decorate(prompt: str, context: Mapping[str, Any]) -> str:
        base = _decorate_with_json_hint(prompt, context)
        return base + "\n\n" + message

    return _decorate


def _set_controller_strategy_id(strategy_id: str) -> ControllerBehavior:
    def _apply(state: MutableMapping[str, Any]) -> None:
        state.setdefault("meta", {})["strategy_id"] = strategy_id

    return _apply


def _toggle_json_mode(state: MutableMapping[str, Any]) -> None:
    state.setdefault("meta", {})["json_only"] = True


# --- Registry ----------------------------------------------------------------

STRATEGY_REGISTRY: Dict[str, StrategyDefinition] = {}


def register_strategy(definition: StrategyDefinition) -> None:
    STRATEGY_REGISTRY[definition.id] = definition


register_strategy(
    StrategyDefinition(
        id="S1",
        name="strict_json",
        description="Deterministic JSON-only handshake with review consensus.",
        json_only=True,
        allow_cot=False,
        max_rounds=8,
        decoding={
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 256,
        },
        consensus_mode="review_handshake",
        pre_round_hooks=(_set_controller_strategy_id("S1"),),
        envelope_validators=(
            _ensure_status,
            _ensure_tag,
        ),
        prompt_decorators=(
            _decorate_with_json_hint,
        ),
        controller_behaviors=(
            _toggle_json_mode,
        ),
        agent_profile=AgentProfile(greedy=True, k_samples=1, max_new_tokens=256),
        metadata={
            "title": "Baseline strict JSON negotiation",
        },
    )
)

register_strategy(
    StrategyDefinition(
        id="S1_QUICK",
        name="strict_json_quick",
        description="Fewer rounds and shorter outputs for quick validation runs.",
        json_only=True,
        allow_cot=False,
        max_rounds=4,
        decoding={
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 192,
        },
        consensus_mode="review_handshake",
        pre_round_hooks=(_set_controller_strategy_id("S1_QUICK"),),
        envelope_validators=(
            _ensure_status,
            _ensure_tag,
        ),
        prompt_decorators=(
            _decorate_with_json_hint,
        ),
        controller_behaviors=(
            _toggle_json_mode,
        ),
        agent_profile=AgentProfile(greedy=True, k_samples=1, max_new_tokens=192),
        metadata={
            "title": "Quick strict JSON runs",
        },
    )
)

register_strategy(
    StrategyDefinition(
        id="S2_PLAN_EXECUTE",
        name="plan_execute",
        description="Planner-to-executor handshake with explicit stage reminders.",
        json_only=True,
        allow_cot=True,
        max_rounds=6,
        decoding={
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 384,
        },
        consensus_mode="review_handshake",
        pre_round_hooks=(
            _set_controller_strategy_id("S2_PLAN_EXECUTE"),
        ),
        envelope_validators=(
            _ensure_status,
            _ensure_tag,
        ),
        prompt_decorators=(
            _decorate_with_guidance(
                "Stage reminder: plan first, then implementation, then testing feedback. Reference the current stage in your JSON tag."
            ),
        ),
        controller_behaviors=(
            _toggle_json_mode,
        ),
        agent_profile=AgentProfile(greedy=True, k_samples=1, max_new_tokens=384),
        metadata={
            "title": "Planner/executor/tester structured turn-taking",
        },
    )
)

register_strategy(
    StrategyDefinition(
        id="S3_SELF_REFINE",
        name="self_refine",
        description="Generator/critic refinement loop with actionable feedback cues.",
        json_only=True,
        allow_cot=True,
        max_rounds=5,
        decoding={
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 320,
        },
        consensus_mode="review_handshake",
        pre_round_hooks=(
            _set_controller_strategy_id("S3_SELF_REFINE"),
        ),
        envelope_validators=(
            _ensure_status,
            _ensure_tag,
        ),
        prompt_decorators=(
            _decorate_with_guidance(
                "Self-refine pattern: proposer shares work, critic responds with issues, proposer revises and highlights changes."
            ),
        ),
        controller_behaviors=(
            _toggle_json_mode,
        ),
        agent_profile=AgentProfile(greedy=True, k_samples=1, max_new_tokens=320),
        metadata={
            "title": "Self-reflection refinement loop",
        },
    )
)

register_strategy(
    StrategyDefinition(
        id="S4_CONSTITUTIONAL",
        name="constitutional_review",
        description="Constitutional critique followed by safe editing and checklist reporting.",
        json_only=True,
        allow_cot=True,
        max_rounds=5,
        decoding={
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 320,
        },
        consensus_mode="review_handshake",
        pre_round_hooks=(
            _set_controller_strategy_id("S4_CONSTITUTIONAL"),
        ),
        envelope_validators=(
            _ensure_status,
            _ensure_tag,
        ),
        prompt_decorators=(
            _decorate_with_guidance(
                "Apply the safety constitution: cite policies, then perform or reject edits with explicit checklist status."
            ),
        ),
        controller_behaviors=(
            _toggle_json_mode,
        ),
        agent_profile=AgentProfile(greedy=True, k_samples=1, max_new_tokens=320),
        metadata={
            "title": "Policy citation and safe-edit workflow",
        },
    )
)

register_strategy(
    StrategyDefinition(
        id="S5_DEBATE",
        name="debate_review",
        description="Peer review dialogue encouraging explicit critiques before convergence.",
        json_only=True,
        allow_cot=True,
        max_rounds=6,
        decoding={
            "do_sample": False,
            "temperature": 0.2,
            "top_p": 0.9,
            "max_new_tokens": 384,
        },
        consensus_mode="review_handshake",
        pre_round_hooks=(
            _set_controller_strategy_id("S5_DEBATE"),
        ),
        envelope_validators=(
            _ensure_status,
            _ensure_tag,
        ),
        prompt_decorators=(
            _decorate_with_guidance(
                "Debate protocol: state your position, critique your partner's reasoning, and document convergence explicitly."
            ),
        ),
        controller_behaviors=(
            _toggle_json_mode,
        ),
        agent_profile=AgentProfile(greedy=False, k_samples=1, max_new_tokens=384),
        metadata={
            "title": "Peer debate with critique tracking",
        },
    )
)
