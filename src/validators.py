from __future__ import annotations

import re
from functools import partial
from typing import Any, Callable, Dict


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.findall(r"[^.!?]+[.!?]*", text) if s.strip()]


def concise_text_validator(
    text: str,
    max_sentences: int | None = None,
    max_tokens: int | None = None,
    strict: bool = False,
) -> str:
    """Trim whitespace and optionally enforce concise responses.

    When ``strict`` is False (default) the validator truncates the text to fit the
    requested limits. When ``strict`` is True the function raises ``ValueError``
    instead of truncating.
    """

    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    sentences = _split_sentences(cleaned)
    if max_sentences and len(sentences) > max_sentences:
        if strict:
            raise ValueError(
                f"Response uses {len(sentences)} sentences; limit is {max_sentences}."
            )
        cleaned = " ".join(sentences[:max_sentences]).strip()
        sentences = _split_sentences(cleaned)

    if max_tokens:
        tokens = cleaned.split()
        if len(tokens) > max_tokens:
            if strict:
                raise ValueError(
                    f"Response uses {len(tokens)} tokens; limit is {max_tokens}."
                )
            cleaned = " ".join(tokens[:max_tokens]).strip()

    return cleaned


_VALIDATORS: Dict[str, Callable[..., str]] = {
    "concise_text": concise_text_validator,
}


def get_validator(name: str, params: Dict[str, Any] | None = None) -> Callable[[str], str]:
    try:
        fn = _VALIDATORS[name]
    except KeyError as exc:
        raise KeyError(f"Validator '{name}' is not registered.") from exc
    if params:
        return partial(fn, **params)
    return fn

