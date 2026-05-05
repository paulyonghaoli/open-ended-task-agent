from __future__ import annotations

import os
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI as _OpenAI
    _openai_class = _OpenAI
except ImportError:
    _openai_class = None  # type: ignore[assignment]

_client: Any = None


def _get_client() -> Any:
    global _client
    if _client is None:
        key = os.environ.get("OPENAI_API_KEY", "")
        _client = _openai_class(api_key=key)
    return _client


def is_openai_available() -> bool:
    return _openai_class is not None and bool(os.environ.get("OPENAI_API_KEY"))


def call_openai(
    prompt: str,
    model: str = "gpt-4o-mini",
    system: str = "You are a precise, concise analytical agent. Be direct and structured.",
    max_tokens: int = 600,
    temperature: float = 0.0,
) -> str:
    if _openai_class is None:
        raise RuntimeError("openai SDK not installed. Run: pip install openai")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set.")
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def safe_generate(
    prompt: str,
    fallback: str,
    model: str = "gpt-4o-mini",
    system: str = "You are a precise, concise analytical agent. Be direct and structured.",
    max_tokens: int = 600,
) -> str:
    try:
        return call_openai(prompt, model=model, system=system, max_tokens=max_tokens)
    except Exception:
        return fallback
