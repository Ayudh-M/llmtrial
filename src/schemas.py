from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any, Union

from jsonschema import Draft7Validator
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any

class FinalSolution(BaseModel):
    canonical_text: str
    sha256: Optional[str] = None
    model_config = ConfigDict(extra="allow")

class Envelope(BaseModel):
    # Allow any bracketed, uppercase-style tag (e.g. [PLAN], [SOLVER]) while still
    # supporting the historical [CONTACT]/[SOLVED] tokens enforced elsewhere.
    tag: str = Field(pattern=r"^\[[A-Z0-9_:-]+\]$")
    status: str
    content: Optional[Dict[str, Any]] = None
    final_solution: Optional[FinalSolution] = None

    def is_solved(self) -> bool:
        return self.tag == "[SOLVED]" and self.status == "SOLVED" and self.final_solution is not None

# Allowed enums (lightweight guard)
ALLOWED_STATUS = {"WORKING","NEED_PEER","PROPOSED","READY_TO_SOLVE","SOLVED"}


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
    """Return a cached jsonschema validator for the given schema reference."""
    if not schema_ref:
        return None
    path = _resolve_schema_path(schema_ref)
    return _load_json_schema(str(path))
