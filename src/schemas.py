from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class FinalSolution(BaseModel):
    canonical_text: str
    sha256: Optional[str] = None

class Envelope(BaseModel):
    tag: str = Field(pattern=r"^\[(CONTACT|SOLVED)\]$")
    status: str
    content: Optional[Dict[str, Any]] = None
    final_solution: Optional[FinalSolution] = None

    def is_solved(self) -> bool:
        return self.tag == "[SOLVED]" and self.status == "SOLVED" and self.final_solution is not None

# Allowed enums (lightweight guard)
ALLOWED_STATUS = {"WORKING","NEED_PEER","PROPOSED","READY_TO_SOLVE","SOLVED"}
