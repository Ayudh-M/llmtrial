
from __future__ import annotations
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
        data = json.loads(path.read_text(encoding="utf-8"))
        return Draft7Validator(data)
    return Draft7Validator(schema)


def validate_envelope(obj: Dict[str, Any], schema: SchemaArg) -> Tuple[bool, list[str]]:
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
