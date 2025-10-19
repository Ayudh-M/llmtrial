
from __future__ import annotations
import json
from jsonschema import validate, Draft7Validator
from typing import Any, Dict, Tuple

def load_schema(schema: Dict[str, Any]) -> Draft7Validator:
    return Draft7Validator(schema)

def validate_envelope(obj: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, list[str]]:
    validator = load_schema(schema)
    errors = sorted(validator.iter_errors(obj), key=lambda e: e.path)
    if errors:
        return False, [f"{'/'.join(map(str,e.path))}: {e.message}" for e in errors]
    return True, []

def coerce_minimal_defaults(obj: Dict[str, Any]) -> Dict[str, Any]:
    o = dict(obj)
    # Minimal coercions to help flaky models
    o.setdefault("status", "WORKING")
    if o.get("status") == "SOLVED" and "final_solution" not in o:
        o["final_solution"] = {"canonical_text": ""}
    return o
