import pytest

from src.model_loader import TINY_MODEL_ID, generate_chat_completion, load_causal_lm


@pytest.mark.integration
def test_tiny_model_loads_and_generates():
    try:
        tokenizer, model = load_causal_lm(TINY_MODEL_ID, dtype="fp32", device_map="cpu")
    except Exception as exc:  # pragma: no cover - depends on environment
        pytest.skip(f"Tiny model unavailable: {exc}")

    messages = [
        {"role": "system", "content": "You are echo."},
        {"role": "user", "content": "Say hello"},
    ]
    output = generate_chat_completion(
        tokenizer,
        model,
        messages,
        decoding={"max_new_tokens": 4, "temperature": 0.0},
    )
    assert isinstance(output, str)
    assert output.strip()
    if getattr(model, "offline_stub", False):
        assert "offline tiny model" in output
