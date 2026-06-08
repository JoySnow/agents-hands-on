"""DeepSeek API client wrapper (OpenAI-compatible).

Reuses the pattern from agents-examples/deepseek_api_connect_validation.py
to call DeepSeek's chat completions API via the openai SDK.
"""

import json
import os
import time
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load .env file from project root (for DEEPSEEK_API_KEY)
load_dotenv()

_client: Optional[OpenAI] = None


def get_llm() -> OpenAI:
    """Get or create the DeepSeek API client singleton.

    Returns:
        Configured OpenAI client pointing at DeepSeek's API.

    Raises:
        ValueError: If the DEEPSEEK_API_KEY environment variable is not set.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY environment variable is not set.\n"
                "Set it with: export DEEPSEEK_API_KEY='sk-xxxx'"
            )
        _client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    return _client


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    model: str = "deepseek-v4-flash",
    temperature: float = 0.0,
    reasoning_effort: str = "low",
    max_retries: int = 2,
    **kwargs: Any,
) -> str:
    """Unified entry point for text generation via DeepSeek API.

    Args:
        system_prompt: System-level instruction for the model.
        user_prompt: User message / query content.
        model: Model identifier (default: deepseek-v4-flash).
        temperature: Sampling temperature (0 = deterministic).
        reasoning_effort: "low", "medium", or "high".
        max_retries: Number of retries on API failure.
        **kwargs: Additional arguments passed to client.chat.completions.create.

    Returns:
        The model's response text.

    Raises:
        RuntimeError: If all retries fail.
    """
    client = get_llm()
    last_error = None

    for attempt in range(1, max_retries + 2):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                stream=False,
                reasoning_effort=reasoning_effort,
                **kwargs,
            )
            text = response.choices[0].message.content
            if text is None:
                text = ""
            return text.strip()

        except Exception as e:
            last_error = e
            if attempt <= max_retries:
                time.sleep(1 * attempt)

    raise RuntimeError(
        f"DeepSeek API call failed after {max_retries + 1} attempts: {last_error}"
    )


def structured_completion(
    system_prompt: str,
    user_prompt: str,
    model: str = "deepseek-v4-flash",
    temperature: float = 0.0,
    max_retries: int = 2,
) -> str:
    """Raw text completion (caller handles JSON parsing externally).

    This is a thin wrapper — use it when the prompt already instructs
    the model to return valid JSON. The caller is responsible for
    json.loads() + Pydantic validation afterwards.

    Why not with_structured_output:
        DeepSeek's OpenAI-compatible API does not support the
        response_format={"type": "json_object", "schema": ...} extension
        that some providers offer. We keep JSON parsing + validation
        in the caller layer, following the pattern from
        guardrail_self_correction_pydantic.py.
    """
    return chat_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=temperature,
        max_retries=max_retries,
    )


def check_api_key() -> bool:
    """Verify that the DEEPSEEK_API_KEY is set and the API is reachable.

    Returns:
        True if the API responds, False otherwise.
    """
    try:
        result = chat_completion(
            "You are a helpful assistant.",
            "Reply with exactly one word: OK",
            temperature=0.0,
        )
        return "ok" in result.lower()
    except Exception:
        return False
