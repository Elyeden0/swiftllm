"""
SwiftLLM — A blazing-fast universal LLM gateway.

Call OpenAI, Anthropic, Gemini, Mistral, Ollama, Groq, Together AI & AWS Bedrock
through one unified API, powered by a Rust core for maximum performance.

Quick start
-----------
>>> from swiftllm import SwiftLLM
>>> llm = SwiftLLM()
>>> llm.add_provider("openai", api_key="sk-...")
>>> resp = llm.completion("gpt-4o-mini", "What is the meaning of life?")
>>> print(resp.content)

One-liner (LiteLLM-style)
--------------------------
>>> from swiftllm import completion
>>> resp = completion("gpt-4o-mini", "Hello!", api_key="sk-...")
>>> print(resp.content)

Async support
-------------
>>> from swiftllm import async_completion
>>> resp = await async_completion("gpt-4o-mini", "Hello!", api_key="sk-...")
>>> print(resp.content)
"""

import asyncio
from functools import partial

from swiftllm._swiftllm import (
    SwiftLLM,
    Message,
    ChatCompletionResponse,
    ToolCall,
    completion,
)


async def async_completion(
    model: str,
    messages,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    tools: list[dict] | None = None,
    tool_choice=None,
) -> ChatCompletionResponse:
    """Async version of completion() — runs the sync call in a thread pool.

    Ideal for use with FastAPI, aiohttp, or any asyncio-based framework.
    The underlying Rust code releases the GIL, so other Python coroutines
    can run concurrently.
    """
    func = partial(
        completion,
        model,
        messages,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        tools=tools,
        tool_choice=tool_choice,
    )
    return await asyncio.to_thread(func)


__all__ = [
    "SwiftLLM",
    "Message",
    "ChatCompletionResponse",
    "ToolCall",
    "completion",
    "async_completion",
]

__version__ = "0.4.0"
