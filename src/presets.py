# -*- coding: utf-8 -*-
from __future__ import annotations

STRATEGIES = {
    "NL": """STYLE: Use concise natural language. Be direct. Avoid fluff. Use short bullet points when helpful.""",

    "JSON_SCHEMA": """STYLE: Reply ONLY as JSON that conforms to:
{
  "answer": string,
  "steps": string[]
}
No prose outside JSON. If unknown, set fields to empty strings.""",

    "PSEUDOCODE": """STYLE: Reply in compact pseudocode using BEGIN/END, IF, FOR, RETURN. Keep lines short.""",

    "KQMLISH": """STYLE: Use symbolic acts, each on separate lines:
(propose :content "...") 
(inform :content "...") 
(ask :content "...")""",

    "EMERGENT_TOY": """STYLE: Compressed tag-speak <T1:...>; <T2:...>. Minimize tokens. No full sentences.""",

    "DSL": """STYLE: Reply strictly in the provided domain-specific language. If impossible, emit DSL_LIMITATION("<reason>")."""
}

ROLESETS = {
    "Planner-Solver": {
        "A": "You are the Planner. Decompose the task into a minimal actionable plan and hand off.",
        "B": "You are the Solver. Execute the plan and produce the final result."
    },
    "Retriever-Synthesizer": {
        "A": "You are the Retriever. List needed facts and their likely sources (no browsing).",
        "B": "You are the Synthesizer. Using the outline, produce the final answer."
    },
    "Moderator-Editor": {
        "A": "You are the Moderator. State constraints, safety, scope, acceptance criteria. Provide a brief checklist.",
        "B": "You are the Editor. Use the checklist to craft/fix the final answer to meet requirements."
    }
}
