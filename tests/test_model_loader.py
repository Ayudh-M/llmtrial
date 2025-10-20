from __future__ import annotations

from typing import List

import torch
from jinja2 import TemplateError

from src.model_loader import build_inputs


class DummyTokenizer:
    """Minimal tokenizer stub to exercise chat template fallbacks."""

    def __init__(self, fail_calls: int) -> None:
        self.chat_template = "dummy-template"
        self.fail_calls = fail_calls
        self.calls: List[List[dict]] = []
        self.return_tensor = torch.arange(4).unsqueeze(0)
        self.pad_token_id = None
        self.last_prompt: str | None = None

    def apply_chat_template(self, messages, return_tensors=None, add_generation_prompt=True):
        self.calls.append([dict(m) for m in messages])
        if len(self.calls) <= self.fail_calls:
            raise TemplateError(
                "After the optional system message, conversation roles must alternate user/assistant/user/assistant/..."
            )
        return self.return_tensor

    def __call__(self, text, return_tensors=None):
        self.last_prompt = text

        class Output:
            def __init__(self, tensor):
                self.input_ids = tensor

        return Output(self.return_tensor)


def _sample_messages():
    return [
        {"role": "system", "content": "SYS1"},
        {"role": "system", "content": "SYS2"},
        {"role": "user", "content": "Hello"},
    ]


def test_build_inputs_retries_with_system_merge():
    tokenizer = DummyTokenizer(fail_calls=1)
    result = build_inputs(tokenizer, _sample_messages())

    assert torch.equal(result, tokenizer.return_tensor)
    assert len(tokenizer.calls) == 2

    # First attempt preserves raw system messages, second merges them into the user turn.
    first_roles = [msg["role"] for msg in tokenizer.calls[0]]
    assert first_roles[0].lower() == "system"

    second_roles = [msg["role"] for msg in tokenizer.calls[1]]
    assert all(role.lower() != "system" for role in second_roles)
    assert "SYS1" in tokenizer.calls[1][0]["content"]
    assert "SYS2" in tokenizer.calls[1][0]["content"]


def test_build_inputs_falls_back_to_plain_prompt_when_template_keeps_failing():
    tokenizer = DummyTokenizer(fail_calls=2)
    result = build_inputs(tokenizer, _sample_messages())

    assert torch.equal(result, tokenizer.return_tensor)
    assert len(tokenizer.calls) == 2  # tried both candidates
    assert tokenizer.last_prompt is not None
    assert "SYS1" in tokenizer.last_prompt
    assert "Hello" in tokenizer.last_prompt
