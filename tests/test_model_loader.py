from __future__ import annotations

import pytest
from types import SimpleNamespace
from typing import List

import torch
from jinja2 import TemplateError

pytestmark = pytest.mark.skip(
    reason="Legacy control-trailer/consensus disabled in simplified fixed-turn runner"
)

from src.control_trailer import CTRL_SUFFIX
from src.model_loader import build_inputs, generate_with_trailer


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


def test_generate_with_trailer_salvages_missing_trailer(monkeypatch):
    base_input = torch.tensor([[100, 101]], dtype=torch.long)

    def fake_build_inputs(_tokenizer, _prompt, add_generation_prompt=True):  # noqa: ARG001
        return base_input

    monkeypatch.setattr("src.model_loader.build_inputs", fake_build_inputs)

    class SalvageTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def encode(self, text, add_special_tokens=False):  # noqa: ARG002
            return [ord(ch) for ch in text]

        def decode(self, tokens, skip_special_tokens=True):  # noqa: ARG002
            chars = []
            for tok in tokens:
                value = int(tok)
                if skip_special_tokens and value in {self.pad_token_id, self.eos_token_id}:
                    continue
                chars.append(chr(value))
            return "".join(chars)

        def __call__(self, text, return_tensors=None):  # noqa: ARG002
            ids = torch.tensor([[ord(ch) for ch in text]], dtype=torch.long)
            return SimpleNamespace(input_ids=ids)

    class SalvageModel:
        def __init__(self) -> None:
            self.calls = 0
            self.device = torch.device("cpu")

        def generate(self, **kwargs):
            input_ids = kwargs["input_ids"]
            if self.calls == 0:
                extra = torch.tensor([[ord(ch) for ch in "Body only"]], dtype=torch.long)
            else:
                trailer = "<<<CTRL{\"tag\":\"[PLAN]\",\"status\":\"PROPOSED\",\"content\":{}}CTRL>>>"
                extra = torch.tensor([[ord(ch) for ch in trailer]], dtype=torch.long)
            self.calls += 1
            return torch.cat([input_ids, extra.to(input_ids.device)], dim=1)

    tokenizer = SalvageTokenizer()
    model = SalvageModel()

    result = generate_with_trailer(
        model,
        tokenizer,
        prompt=[],
        max_new_tokens=16,
        body_budget=8,
        trailer_budget=8,
        salvage_max_new_tokens=32,
        temperature=0.6,
        top_p=0.9,
    )

    assert "Body only" in result.text
    assert result.text.rstrip().endswith(CTRL_SUFFIX)
    assert "<<<CTRL" in result.text
    assert result.stop_reason == "suffix"
    assert result.tokens_reserved == 16 + 32
    assert result.tokens_used == len("Body only") + len(
        "<<<CTRL{\"tag\":\"[PLAN]\",\"status\":\"PROPOSED\",\"content\":{}}CTRL>>>"
    )
    assert model.calls >= 2
