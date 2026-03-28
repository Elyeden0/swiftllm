"""
SwiftLLM — A blazing-fast universal LLM gateway.

Call OpenAI, Anthropic, Gemini, Mistral & Ollama through one unified API,
powered by a Rust core for maximum performance.

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
"""

from swiftllm._swiftllm import (
    SwiftLLM,
    Message,
    ChatCompletionResponse,
    completion,
)

__all__ = [
    "SwiftLLM",
    "Message",
    "ChatCompletionResponse",
    "completion",
]

__version__ = "0.3.0"
