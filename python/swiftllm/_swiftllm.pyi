"""Type stubs for the native Rust extension."""

from typing import Optional, Union


class Message:
    """A single message in a chat conversation."""
    role: str
    content: Optional[str]

    def __init__(self, role: str, content: Optional[str] = None) -> None: ...


class ToolCall:
    """A tool call returned by the model."""
    id: str
    function_name: str
    function_arguments: str


class ChatCompletionResponse:
    """Response from a chat completion call."""
    id: str
    model: str
    content: Optional[str]
    finish_reason: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    tool_calls: Optional[list[ToolCall]]

    @property
    def raw(self) -> dict:
        """The full raw response as a Python dict."""
        ...


class SwiftLLM:
    """
    The main SwiftLLM gateway client.

    Parameters
    ----------
    cache : bool
        Enable response caching (default True).
    cache_max_size : int
        Maximum cached responses (default 1000).
    cache_ttl : int
        Cache TTL in seconds (default 300).

    Supported providers: openai, anthropic, gemini, mistral, ollama,
    groq, together, bedrock.
    """

    def __init__(
        self,
        *,
        cache: bool = True,
        cache_max_size: int = 1000,
        cache_ttl: int = 300,
    ) -> None: ...

    def add_provider(
        self,
        name: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        models: Optional[list[str]] = None,
        priority: int = 100,
    ) -> None:
        """Register a provider (openai, anthropic, gemini, mistral, ollama, groq, together, bedrock)."""
        ...

    def set_key(self, provider: str, api_key: str) -> None:
        """Set an API key for a provider. Auto-registers the provider if needed."""
        ...

    def completion(
        self,
        model: str,
        messages: Union[str, list[dict[str, str]], list[Message]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
    ) -> ChatCompletionResponse:
        """
        Send a chat completion request.

        Parameters
        ----------
        model : str
            Model identifier (e.g. "gpt-4o-mini", "claude-sonnet-4-20250514").
        messages : str | list[dict] | list[Message]
            A plain string (becomes a user message), or a list of message dicts/objects.
        temperature : float, optional
        max_tokens : int, optional
        top_p : float, optional
        tools : list[dict], optional
            Tool definitions for function calling (OpenAI format).
        tool_choice : str | dict, optional
            Controls tool usage: "auto", "none", "required", or a dict.
        """
        ...

    def list_providers(self) -> list[str]:
        """Return names of all configured providers."""
        ...

    def list_models(self) -> list[dict[str, str]]:
        """Return all models across all configured providers."""
        ...

    def serve(self, *, port: int = 8080) -> None:
        """Start the SwiftLLM HTTP gateway server (blocking)."""
        ...


def completion(
    model: str,
    messages: Union[str, list[dict[str, str]], list[Message]],
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    tools: Optional[list[dict]] = None,
    tool_choice: Optional[Union[str, dict]] = None,
) -> ChatCompletionResponse:
    """
    Quick one-shot completion (LiteLLM-style).

    Provider is auto-detected from the model name.
    Supports 8 providers: OpenAI, Anthropic, Gemini, Mistral, Ollama,
    Groq, Together AI, and AWS Bedrock.
    """
    ...
