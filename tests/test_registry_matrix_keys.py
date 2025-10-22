from pathlib import Path

import pytest
import yaml


_REGISTRY = yaml.safe_load(Path("prompts/registry.yaml").read_text(encoding="utf-8"))["scenarios"]

_STYLES = ["DSL", "JSON_SCHEMA", "PSEUDOCODE", "KQMLISH", "EMERGENT_TOY", "NL"]
_MATRIX = [
    ("boolean_eval_small", "Boolean-ProposeCheck"),
    ("math_sum_1k", "Math-SolverChecker"),
    ("math_linear_pair", "Math-SolverChecker"),
    ("regex_email_basic", "Regex-AuthorTester"),
    ("sql_2024_amount_per_customer", "SQL-AuthorAuditor"),
    ("entity_normalization_tiny", "Entity-MapperQA"),
    ("headline_xsum_tiny", "Headline-WriterAuditor"),
    ("mcq_arc_tiny", "MCQ-ReasonerAuditor"),
    ("paraphrase_tiny", "Paraphrase-LabelerAuditor"),
    ("translate_en_de_glossary", "Translate-QE"),
    ("winogrande_tiny", "Winogrande-ResolverRefuter"),
    ("writer_physicist_tiny", "Writer-Physicist"),
]


@pytest.mark.parametrize("dataset,pair", _MATRIX)
@pytest.mark.parametrize("style", _STYLES)
def test_matrix_key_present(dataset, style, pair):
    key = f"{dataset}:{style}:{pair}"
    assert key in _REGISTRY, f"Missing scenario key: {key}"
