from __future__ import annotations

"""Domain-specific language (DSL) support for agent envelopes."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .utils import sha256_hex

# Canonical grammar description extracted from the protocol specification.
_BASE_GRAMMAR = """
Envelope ::= {
  "role": STRING,
  "domain": STRING,
  "task_understanding": STRING,
  "public_message": STRING,
  "artifact": Artifact,
  "needs_from_peer": Needs,
  "handoff_to": STRING,
  "status": Status,
  "tag": Tag,
  "content": OBJECT?,
  "final_solution": FinalSolution?
}
Artifact ::= {
  "type": ArtifactType,
  "content": OBJECT
}
Needs ::= LIST (STRING)   ; up to three concise asks
FinalSolution ::= { "canonical_text": STRING, "sha256": STRING? }
ArtifactType ::= "component_spec" | "code_patch" | "outline" | "fact_pack" | "source_pack" | "plan" | "dataset" | "results"
Status ::= "WORKING" | "NEED_PEER" | "PROPOSED" | "READY_TO_SOLVE" | "REVISED" | "SOLVED"
Tag ::= "[CONTACT]" | "[SOLVED]"
""".strip()

# Keywords referenced in prompts and enforcement logic.
_BASE_KEYWORDS = [
    "role",
    "domain",
    "task_understanding",
    "public_message",
    "artifact",
    "artifact.type",
    "artifact.content",
    "needs_from_peer",
    "handoff_to",
    "status",
    "tag",
    "final_solution",
    "final_solution.canonical_text",
]

_ALLOWED_TAGS = ("[CONTACT]", "[SOLVED]")
_ALLOWED_STATUS = ("WORKING", "NEED_PEER", "PROPOSED", "READY_TO_SOLVE", "REVISED", "SOLVED")
_ARTIFACT_TYPES = (
    "component_spec",
    "code_patch",
    "outline",
    "fact_pack",
    "source_pack",
    "plan",
    "dataset",
    "results",
)


@dataclass
class DSLExtension:
    """Scenario-provided productions merged into the base grammar."""

    productions: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    artifact_types: List[str] = field(default_factory=list)
    artifact_content_rules: Dict[str, Sequence[str]] = field(default_factory=dict)

    def normalized(self) -> "DSLExtension":
        prods = [str(p).strip() for p in self.productions if str(p).strip()]
        kws = []
        seen_kw = set()
        for kw in self.keywords:
            kw_s = str(kw).strip()
            if kw_s and kw_s not in seen_kw:
                seen_kw.add(kw_s)
                kws.append(kw_s)
        art_types = []
        seen_art = set()
        for typ in self.artifact_types:
            t = str(typ).strip()
            if t and t not in seen_art:
                seen_art.add(t)
                art_types.append(t)
        rules: Dict[str, Sequence[str]] = {}
        for key, values in (self.artifact_content_rules or {}).items():
            key_s = str(key).strip()
            if not key_s:
                continue
            if isinstance(values, (list, tuple, set)):
                unique_vals: List[str] = []
                seen_val = set()
                for v in values:
                    v_s = str(v).strip()
                    if v_s and v_s not in seen_val:
                        seen_val.add(v_s)
                        unique_vals.append(v_s)
                rules[key_s] = unique_vals
            else:
                v_s = str(values).strip()
                if v_s:
                    rules[key_s] = [v_s]
        return DSLExtension(prods, kws, art_types, rules)


@dataclass
class DSLSpec:
    """Static specification for the DSL."""

    grammar: str
    keywords: List[str]
    artifact_types: List[str]
    allowed_status: List[str]
    allowed_tags: List[str]
    artifact_content_rules: Dict[str, Sequence[str]] = field(default_factory=dict)

    def create_validator(self, extension: Optional[DSLExtension] = None) -> "DSLValidator":
        ext = extension.normalized() if extension else None
        grammar = self.grammar
        if ext and ext.productions:
            grammar = grammar.rstrip() + "\n" + "\n".join(ext.productions)
        keywords = list(self.keywords)
        if ext and ext.keywords:
            for kw in ext.keywords:
                if kw not in keywords:
                    keywords.append(kw)
        artifact_types: List[str] = list(dict.fromkeys(self.artifact_types))
        if ext and ext.artifact_types:
            for typ in ext.artifact_types:
                if typ not in artifact_types:
                    artifact_types.append(typ)
        rules = dict(self.artifact_content_rules)
        if ext and ext.artifact_content_rules:
            rules.update(ext.artifact_content_rules)
        return DSLValidator(
            grammar=grammar,
            keywords=keywords,
            allowed_status=self.allowed_status,
            allowed_tags=self.allowed_tags,
            artifact_types=artifact_types,
            artifact_content_rules=rules,
        )


def default_dsl_spec() -> DSLSpec:
    return DSLSpec(
        grammar=_BASE_GRAMMAR,
        keywords=list(_BASE_KEYWORDS),
        artifact_types=list(_ARTIFACT_TYPES),
        allowed_status=list(_ALLOWED_STATUS),
        allowed_tags=list(_ALLOWED_TAGS),
    )


def extension_from_config(cfg: Optional[Dict[str, Any]]) -> Optional[DSLExtension]:
    if not cfg:
        return None
    productions = cfg.get("productions") if isinstance(cfg, dict) else None
    keywords = cfg.get("keywords") if isinstance(cfg, dict) else None
    artifact_types = cfg.get("artifact_types") if isinstance(cfg, dict) else None
    content_rules = cfg.get("artifact_content_rules") if isinstance(cfg, dict) else None
    ext = DSLExtension(
        productions=list(productions or []),
        keywords=list(keywords or []),
        artifact_types=list(artifact_types or []),
        artifact_content_rules=dict(content_rules or {}),
    )
    normalized = ext.normalized()
    if (
        not normalized.productions
        and not normalized.keywords
        and not normalized.artifact_types
        and not normalized.artifact_content_rules
    ):
        return None
    return normalized


class DSLValidationError(ValueError):
    def __init__(self, errors: Sequence[str], envelope: Optional[Dict[str, Any]] = None):
        super().__init__("; ".join(errors))
        self.errors = list(errors)
        self.envelope = envelope


@dataclass
class DSLParseResult:
    status: str
    tag: str
    artifact_type: Optional[str]
    canonical_text: Optional[str]
    keywords_used: List[str]
    needs_from_peer: List[str]
    public_message: str
    public_tags: List[str]
    grammar_sha256: str

    def to_trace_entry(self, round_index: int, actor: str) -> Dict[str, Any]:
        return {
            "round": round_index,
            "actor": actor,
            "status": self.status,
            "tag": self.tag,
            "artifact_type": self.artifact_type,
            "canonical_text": self.canonical_text,
            "keywords_used": list(self.keywords_used),
            "needs_from_peer": list(self.needs_from_peer),
            "public_tags": list(self.public_tags),
            "public_message": self.public_message,
            "grammar_sha256": self.grammar_sha256,
        }


class DSLValidator:
    def __init__(
        self,
        *,
        grammar: str,
        keywords: Sequence[str],
        allowed_status: Sequence[str],
        allowed_tags: Sequence[str],
        artifact_types: Sequence[str],
        artifact_content_rules: Optional[Dict[str, Sequence[str]]] = None,
    ) -> None:
        self.grammar = grammar
        self.keywords = list(dict.fromkeys(keywords))
        self.allowed_status = {s.upper(): s.upper() for s in allowed_status}
        self.allowed_tags = {t: t for t in allowed_tags}
        self.artifact_types = {t: t for t in artifact_types}
        self.artifact_content_rules = artifact_content_rules or {}
        self.grammar_sha256 = sha256_hex(grammar)

    # -- helpers -----------------------------------------------------------------

    def _keyword_present(self, envelope: Dict[str, Any], keyword: str) -> bool:
        parts = keyword.split(".")
        current: Any = envelope
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False
        return True

    def _collect_keywords(self, envelope: Dict[str, Any]) -> List[str]:
        present = []
        for kw in self.keywords:
            if self._keyword_present(envelope, kw):
                present.append(kw)
        return present

    def _collect_public_tags(self, public_message: str) -> List[str]:
        tags = []
        for tag in self.allowed_tags:
            if tag in public_message:
                tags.append(tag)
        return tags

    # -- validation ---------------------------------------------------------------

    def validate(self, envelope: Dict[str, Any]) -> DSLParseResult:
        if not isinstance(envelope, dict):
            raise DSLValidationError(["Envelope must be a JSON object"], envelope=envelope)

        errors: List[str] = []

        def require(condition: bool, message: str) -> None:
            if not condition:
                errors.append(message)

        role = envelope.get("role")
        domain = envelope.get("domain")
        task_understanding = envelope.get("task_understanding")
        public_message = envelope.get("public_message")
        artifact = envelope.get("artifact")
        needs = envelope.get("needs_from_peer")
        handoff = envelope.get("handoff_to")
        status_raw = envelope.get("status")
        tag_raw = envelope.get("tag")
        final_solution = envelope.get("final_solution")

        require(isinstance(role, str) and role.strip(), "role must be a non-empty string")
        require(isinstance(domain, str) and domain.strip(), "domain must be a non-empty string")
        require(
            isinstance(task_understanding, str) and task_understanding.strip(),
            "task_understanding must be a non-empty string",
        )
        require(isinstance(public_message, str) and public_message.strip(), "public_message must be a non-empty string")
        require(isinstance(artifact, dict), "artifact must be an object with type/content")
        require(isinstance(needs, list), "needs_from_peer must be a list")
        require(isinstance(handoff, str) and handoff.strip(), "handoff_to must be a non-empty string")

        status = str(status_raw or "").upper()
        tag = str(tag_raw or "").strip()
        require(status in self.allowed_status, f"status '{status_raw}' is not allowed by grammar")
        require(tag in self.allowed_tags, f"tag '{tag_raw}' is not allowed by grammar")

        artifact_type: Optional[str] = None
        if isinstance(artifact, dict):
            artifact_type = artifact.get("type")
            require(isinstance(artifact_type, str) and artifact_type.strip(), "artifact.type must be a string")
            if isinstance(artifact_type, str) and artifact_type.strip():
                typ_norm = artifact_type.strip()
                require(
                    typ_norm in self.artifact_types,
                    f"artifact.type '{artifact_type}' is not part of the DSL grammar",
                )
                required_keys = self.artifact_content_rules.get(typ_norm, [])
                content = artifact.get("content") if isinstance(artifact, dict) else None
                require(isinstance(content, dict), "artifact.content must be an object")
                if isinstance(content, dict):
                    for key in required_keys:
                        require(key in content, f"artifact.content missing required key '{key}' for type '{typ_norm}'")
        else:
            require(False, "artifact must be provided as an object")

        needs_list: List[str] = []
        if isinstance(needs, list):
            if len(needs) > 3:
                errors.append("needs_from_peer may include at most three entries")
            for idx, entry in enumerate(needs):
                if not isinstance(entry, str) or not entry.strip():
                    errors.append(f"needs_from_peer[{idx}] must be a non-empty string")
                else:
                    needs_list.append(entry)

        if status == "SOLVED":
            require(tag == "[SOLVED]", "SOLVED status requires tag [SOLVED]")
        if tag == "[SOLVED]":
            require(status == "SOLVED", "tag [SOLVED] requires status SOLVED")
        if status == "NEED_PEER":
            require("[CONTACT]" in public_message, "NEED_PEER status requires [CONTACT] in public_message")
        if status == "SOLVED":
            require("[SOLVED]" in public_message, "SOLVED status requires [SOLVED] in public_message")

        canonical_text: Optional[str] = None
        if final_solution is not None:
            require(isinstance(final_solution, dict), "final_solution must be an object when present")
            if isinstance(final_solution, dict):
                canonical = final_solution.get("canonical_text")
                if canonical is not None:
                    require(isinstance(canonical, str) and canonical.strip(), "final_solution.canonical_text must be a non-empty string")
                    if isinstance(canonical, str) and canonical.strip():
                        canonical_text = canonical.strip()
                elif status == "SOLVED":
                    errors.append("final_solution.canonical_text is required when status is SOLVED")
        elif status == "SOLVED":
            errors.append("final_solution must be provided when status is SOLVED")

        if errors:
            raise DSLValidationError(errors, envelope=envelope)

        keywords_used = self._collect_keywords(envelope)
        public_tags = self._collect_public_tags(public_message)

        return DSLParseResult(
            status=status,
            tag=tag,
            artifact_type=artifact_type if isinstance(artifact_type, str) else None,
            canonical_text=canonical_text,
            keywords_used=keywords_used,
            needs_from_peer=needs_list,
            public_message=public_message,
            public_tags=public_tags,
            grammar_sha256=self.grammar_sha256,
        )


__all__ = [
    "DSLParseResult",
    "DSLSpec",
    "DSLValidationError",
    "DSLValidator",
    "DSLExtension",
    "default_dsl_spec",
    "extension_from_config",
]
