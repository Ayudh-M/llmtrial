from __future__ import annotations

"""Utility wrappers around jsonschema validation and pydantic models."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any, Union

from jsonschema import Draft7Validator
from pydantic import BaseModel, Field, ConfigDict

ROOT = Path(__file__).resolve().parents[1]


class FinalSolution(BaseModel):
    canonical_text: str
    sha256: Optional[str] = None
    model_config = ConfigDict(extra="allow")


class Envelope(BaseModel):
    tag: str = Field(pattern=r"^\[(CONTACT|SOLVED)\]$")
    status: str
    content: Optional[Dict[str, Any]] = None
    final_solution: Optional[FinalSolution] = None

    model_config = ConfigDict(extra="allow")

    def is_solved(self) -> bool:
        return (
            self.tag == "[SOLVED]"
            and self.status == "SOLVED"
            and self.final_solution is not None
            and bool(self.final_solution.canonical_text)
        )


SchemaLike = Union[str, Path]


def _resolve_schema_path(schema_ref: SchemaLike) -> Path:
    path = Path(schema_ref)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


@lru_cache(maxsize=None)
def _load_json_schema(schema_path: str) -> Draft7Validator:
    path = Path(schema_path)
    if not path.exists():
        raise FileNotFoundError(f"Schema not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return Draft7Validator(data)


def get_envelope_validator(schema_ref: Optional[SchemaLike]) -> Optional[Draft7Validator]:
    if not schema_ref:
        return None
    path = _resolve_schema_path(schema_ref)
    return _load_json_schema(str(path))
