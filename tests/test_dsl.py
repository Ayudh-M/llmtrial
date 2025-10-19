import pytest

from src.dsl import default_dsl_spec, DSLExtension, DSLValidationError


def _make_envelope(artifact_type: str = "results"):
    return {
        "role": "Mock",
        "domain": "demo",
        "task_understanding": "demo task",
        "public_message": "[CONTACT] initial",
        "artifact": {"type": artifact_type, "content": {"proposal": "value"}},
        "needs_from_peer": ["confirm"],
        "handoff_to": "peer",
        "status": "PROPOSED",
        "tag": "[CONTACT]",
        "content": {"note": "draft"},
    }


def test_validator_accepts_valid_envelope():
    spec = default_dsl_spec()
    validator = spec.create_validator()
    env = _make_envelope()
    parsed = validator.validate(env)
    assert parsed.status == "PROPOSED"
    assert "artifact.type" in parsed.keywords_used
    assert parsed.canonical_text is None


def test_validator_rejects_missing_artifact():
    spec = default_dsl_spec()
    validator = spec.create_validator()
    env = _make_envelope()
    env.pop("artifact")
    with pytest.raises(DSLValidationError):
        validator.validate(env)


def test_domain_extension_merging():
    spec = default_dsl_spec()
    extension = DSLExtension(
        productions=['ArtifactType ::= "engineering_component"'],
        artifact_types=["engineering_component"],
        keywords=["artifact.components"],
        artifact_content_rules={"engineering_component": ["components"]},
    )
    validator = spec.create_validator(extension)
    env = _make_envelope("engineering_component")
    env["artifact"]["content"] = {"components": ["motor"]}
    parsed = validator.validate(env)
    assert parsed.artifact_type == "engineering_component"
