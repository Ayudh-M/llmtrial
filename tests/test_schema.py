from src.json_enforcer import coerce_minimal_defaults, validate_envelope
from src.schemas import get_envelope_validator

SCHEMA = get_envelope_validator("schemas/envelope.schema.json")


def test_validate_envelope_requires_status():
    ok, errors = validate_envelope({}, SCHEMA)
    assert not ok
    assert errors


def test_coerce_defaults_inserts_keys():
    base = {"status": "SOLVED"}
    coerced = coerce_minimal_defaults(base)
    assert coerced["tag"] == "[SOLVED]"
    assert coerced["final_solution"]["canonical_text"] == ""


def test_validate_envelope_success():
    env = {
        "status": "SOLVED",
        "tag": "[SOLVED]",
        "content": {"acl": "PROPOSE: hi"},
        "final_solution": {"canonical_text": "42"},
    }
    ok, errors = validate_envelope(env, SCHEMA)
    assert ok, errors
