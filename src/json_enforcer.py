from __future__ import annotations

"""JSON schema helpers used to validate agent envelopes."""

import json
from pathlib import Path
from typing import Any, Dict, Tuple, Union

from jsonschema import Draft7Validator

SchemaArg = Union[Draft7Validator, Dict[str, Any], str, Path]


def load_schema(schema: SchemaArg) -> Draft7Validator:
    if isinstance(schema, Draft7Validator):
        return schema
    if isinstance(schema, (str, Path)):
        path = Path(schema)
        if not path.exists():
            raise FileNotFoundError(f"Schema not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return Draft7Validator(data)
    if isinstance(schema, dict):
        return Draft7Validator(schema)
    raise TypeError("Unsupported schema argument type.")


def validate_envelope(obj: Dict[str, Any], schema: SchemaArg) -> Tuple[bool, list[str]]:
    validator = load_schema(schema)
    errors = sorted(validator.iter_errors(obj), key=lambda err: err.path)
    if errors:
        formatted = [f"{'/'.join(map(str, err.path))}: {err.message}" for err in errors]
        return False, formatted
    return True, []


def coerce_minimal_defaults(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in a few protocol defaults to help flaky model outputs."""

    coerced = dict(obj)
    coerced.setdefault("status", "WORKING")
    coerced.setdefault("tag", "[CONTACT]" if coerced["status"] != "SOLVED" else "[SOLVED]")
    coerced.setdefault("needs_from_peer", [])
    coerced.setdefault(
        "artifact",
        {"type": "results", "content": {}},
    )
    if coerced.get("status") == "SOLVED":
        final = coerced.setdefault("final_solution", {})
        final.setdefault("canonical_text", "")
    coerced.setdefault("content", {})
    if isinstance(coerced["content"], dict):
        coerced["content"].setdefault("acl", "PROPOSE: pending clarification")
    return coerced
