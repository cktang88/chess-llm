"""OpenRouter client + generic chat helper with budget charging."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

from .budget import Budget

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-5.4-mini"
DEFAULT_REASONING_EFFORT = "low"


def make_client() -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set (check .env)")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)


def chat(
    *,
    messages: list[dict],
    budget: Budget,
    tag: str,
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[str, dict]:
    """Returns (content_text, meta). Charges budget with OpenRouter-reported cost.

    Note on max_tokens: OpenRouter reserves credits up-front based on the
    call's max_output_tokens (model default ≈ 65536 for gpt-5.4). Passing a
    tighter max_tokens (e.g. 8000 for reflect) avoids 402 "insufficient
    credits" errors when the actual output is far smaller than the default.
    """
    client = client or make_client()
    kwargs: dict = dict(model=model, messages=messages)
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = max_tokens
    resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content or ""
    usage = resp.usage
    in_toks = getattr(usage, "prompt_tokens", 0) or 0
    out_toks = getattr(usage, "completion_tokens", 0) or 0
    reported = getattr(usage, "cost", None)  # OpenRouter-specific field
    budget.charge(
        tag=tag, model=model,
        input_tokens=in_toks, output_tokens=out_toks,
        reported_cost=reported,
    )
    meta = {"input_tokens": in_toks, "output_tokens": out_toks,
            "cost": reported if reported is not None else "estimated",
            "model": getattr(resp, "model", model)}
    return content, meta
