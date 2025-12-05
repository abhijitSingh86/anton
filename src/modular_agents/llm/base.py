"""Abstract base for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    from modular_agents.tools.base import ToolCall


@dataclass
class LLMMessage:
    """A message in a conversation."""
    
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM."""
    
    content: str
    model: str
    usage: dict = field(default_factory=dict)  # tokens used
    raw: dict = field(default_factory=dict)  # raw response for debugging


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""
    
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    api_key: str | None = None
    base_url: str | None = None
    extra: dict = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'claude', 'openai', 'ollama')."""
        ...
    
    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        system: str | None = None,
    ) -> LLMResponse:
        """Generate a completion for the given messages."""
        ...
    
    @abstractmethod
    async def stream(
        self,
        messages: list[LLMMessage],
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion for the given messages."""
        ...

    @abstractmethod
    async def chat_with_tools(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
        max_tool_rounds: int = 5,
    ) -> tuple[str | None, list[ToolCall]]:
        """Chat with tool calling support.

        Args:
            messages: Conversation history in provider format
            system: System prompt
            tools: Available tools in provider format
            max_tool_rounds: Maximum tool calling iterations

        Returns:
            Tuple of (final_response, tool_calls_made)
            - If response is None, tool_calls will be populated
            - If response is text, it's the final answer
        """
        ...

    def format_messages(
        self,
        messages: list[LLMMessage],
        system: str | None = None,
    ) -> list[dict]:
        """Format messages for the provider's API."""
        formatted = []
        if system:
            formatted.append({"role": "system", "content": system})
        for msg in messages:
            formatted.append({"role": msg.role, "content": msg.content})
        return formatted


class LLMProviderRegistry:
    """Registry for LLM providers."""
    
    _providers: dict[str, type[LLMProvider]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a provider."""
        def decorator(provider_cls: type[LLMProvider]):
            cls._providers[name] = provider_cls
            return provider_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> type[LLMProvider] | None:
        """Get a provider by name."""
        return cls._providers.get(name)
    
    @classmethod
    def available(cls) -> list[str]:
        """List available providers."""
        return list(cls._providers.keys())
    
    @classmethod
    def create(cls, name: str, config: LLMConfig) -> LLMProvider:
        """Create a provider instance."""
        provider_cls = cls.get(name)
        if not provider_cls:
            raise ValueError(
                f"Unknown LLM provider: {name}. "
                f"Available: {cls.available()}"
            )
        return provider_cls(config)
