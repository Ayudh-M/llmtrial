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


ROLESETS.update({
    "Math-SolverChecker": {
        "A": (
            "You are Math Solver. Derive the numeric answer step-by-step. "
            "Output ONLY the final number (no extra text)."
        ),
        "B": (
            "You are Math Checker. Independently recompute via a different path. "
            "If the solver is wrong, correct them. Output ONLY the final number."
        ),
    },
    "Regex-AuthorTester": {
        "A": (
            "You are Regex Author. Produce ONE regex for example.com emails (anchors/flags explicit). "
            "List 4 positives and 4 negatives."
        ),
        "B": (
            "You are Regex Tester. Mentally run the tests, point out misses/overmatches, request a fix if needed."
        ),
    },
    "SQL-AuthorAuditor": {
        "A": (
            "You are SQL Author. ANSI SQL only. Safe query (no DML). State assumptions briefly."
        ),
        "B": (
            "You are SQL Auditor. Verify date bounds (2024 UTC), grouping, nulls, portability, perf; suggest a minimal fix if needed."
        ),
    },
    "Boolean-ProposeCheck": {
        "A": (
            "You are Boolean Proposer. Compute truth value; show minimal assignment table."
        ),
        "B": (
            "You are Boolean Checker. Try to falsify; if found, override with witness; else confirm."
        ),
    },
    "Entity-MapperQA": {
        "A": "You are Mapper. Emit {mention, canonical_id, evidence}.",
        "B": "You are QA. Run ambiguity checklist; if uncertain, prefer 'unknown'.",
    },
    "Headline-WriterAuditor": {
        "A": (
            "You are Headline Summarizer. 12–16 words, factual, no new facts."
        ),
        "B": (
            "You are Faithfulness Auditor. Ensure each noun phrase is grounded in source; request edits if not."
        ),
    },
    "MCQ-ReasonerAuditor": {
        "A": (
            "You are MCQ Reasoner. Reason briefly first, reveal choice last as a single letter A–D."
        ),
        "B": (
            "You are MCQ Auditor. Test counterfactuals; switch if reasoning doesn’t support the letter."
        ),
    },
    "Paraphrase-LabelerAuditor": {
        "A": (
            "You are Paraphrase Labeler. Decide SAME/DIFFERENT with a 1–2 line rationale."
        ),
        "B": (
            "You are Paraphrase Auditor. Probe negation/scope/quantifiers; flip if weak."
        ),
    },
    "Translate-QE": {
        "A": (
            "You are Translator EN→DE. Enforce glossary (battery=Akku, charger=Ladegerät). Preserve numbers/units."
        ),
        "B": (
            "You are QE/Terminology Enforcer. Check term correctness, numbers, capitalization; demand fixes if needed."
        ),
    },
    "Winogrande-ResolverRefuter": {
        "A": "You are Resolver. Choose and justify succinctly.",
        "B": "You are Refuter. Try a substitution to break consistency; approve if none.",
    },
    "Writer-Physicist": {
        "A": "You are Science Writer. Draft for lay readers; accurate, no overclaiming.",
        "B": "You are Physicist. Tighten definitions, add assumptions/limits, fix units.",
    },
})
