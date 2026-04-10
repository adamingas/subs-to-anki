#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai>=1.75.0",
#   "python-dotenv>=1.0.1",
#   "typer>=0.16.0",
# ]
# ///

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from openai import AsyncOpenAI

app = typer.Typer(no_args_is_help=True, add_completion=False)

ROOT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ROOT_ENV_PATH)

DEFAULT_BASE_URL = os.environ.get("BASE_URL") or os.environ.get("OPENAI_BASE_URL")
DEFAULT_TOKEN = os.environ.get("OPENAI_TOKEN") or os.environ.get("OPENAI_API_KEY")
DEFAULT_MODEL = "qwen/qwen3.5-122b-a10b"


def usage_to_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return json.loads(json.dumps(usage, default=str))


def message_to_dict(message: Any) -> dict[str, Any]:
    model_extra = getattr(message, "model_extra", None) or {}
    return {
        "content": getattr(message, "content", None),
        "reasoning": getattr(message, "reasoning", None),
        "reasoning_content": getattr(message, "reasoning_content", None),
        "model_extra": model_extra,
    }


async def run_probe(
    *,
    client: AsyncOpenAI,
    model: str,
    effort: str | None,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Answer briefly with just the result."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        # reasoning_effort = "mini",
        extra_body={"reasoning": {"effort": "none"}},
    )
    message = completion.choices[0].message
    return {
        "effort": effort,
        "message": message_to_dict(message),
        "usage": usage_to_dict(completion.usage),
    }


@app.command()
def main(
    model: str = typer.Option(DEFAULT_MODEL, "--model"),
    base_url: str = typer.Option(DEFAULT_BASE_URL or "", "--base-url"),
    api_key: str = typer.Option(DEFAULT_TOKEN or "", "--token"),
    prompt: str = typer.Option("What is 2+2?", "--prompt"),
    max_tokens: int = typer.Option(64, "--max-tokens"),
    temperature: float = typer.Option(0.2, "--temperature"),
) -> None:
    if not base_url:
        raise typer.BadParameter("Missing base URL. Set BASE_URL in .env or pass --base-url.")
    if not api_key:
        raise typer.BadParameter("Missing API key. Set OPENAI_TOKEN in .env or pass --token.")

    async def runner() -> None:
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        none_result = await run_probe(
            client=client,
            model=model,
            effort=None,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        low_result = await run_probe(
            client=client,
            model=model,
            effort="low",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        print(json.dumps({"none": none_result, "low": low_result}, ensure_ascii=False, indent=2))

    asyncio.run(runner())


if __name__ == "__main__":
    app()
