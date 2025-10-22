from pathlib import Path


_REGISTRY_TEXT = Path("prompts/registry.yaml").read_text(encoding="utf-8")


def test_no_inner_default_model_anchor_redefinition():
    assert "\n    models: &default_matrix_models" not in _REGISTRY_TEXT, (
        "Do not re-anchor default models inside a scenario; use '*default_matrix_models'"
    )

    header = _REGISTRY_TEXT.split("scenarios:")[0]
    assert "&default_matrix_models" in header, "defaults.models anchor missing in header"
