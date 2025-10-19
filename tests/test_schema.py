
from src.json_enforcer import validate_envelope, coerce_minimal_defaults
from src.schemas import get_envelope_validator

SCHEMA_VALIDATOR = get_envelope_validator("schemas/envelope.schema.json")

def test_envelope_validation_happy():
    obj = {
        "status":"SOLVED",
        "tag":"[SOLVED]",
        "content": {"acl": "SOLVED: returning final answer"},
        "final_solution":{"canonical_text":"42"}
    }
    ok, errs = validate_envelope(obj, SCHEMA_VALIDATOR)
    assert ok, errs

def test_envelope_missing_solution_rejected():
    obj = {"status":"SOLVED"}
    ok, errs = validate_envelope(obj, SCHEMA_VALIDATOR)
    assert not ok and errs

def test_coerce_then_validate():
    bad = {"status":"SOLVED"}
    fixed = coerce_minimal_defaults(bad)
    fixed.setdefault("tag", "[SOLVED]")
    ok, errs = validate_envelope(fixed, SCHEMA_VALIDATOR)
    assert not ok  # still missing canonical_text
    ok, errs = validate_envelope(fixed, SCHEMA)
    assert ok
    fixed["final_solution"]["canonical_text"] = "X"
    ok2, errs2 = validate_envelope(fixed, SCHEMA_VALIDATOR)
    assert ok2
