"""Ollama provider for local LLMs."""

from __future__ import annotations

from typing import AsyncIterator

from .base import LLMConfig, LLMMessage, LLMProvider, LLMProviderRegistry, LLMResponse
from modular_agents.tools.base import ToolCall


@LLMProviderRegistry.register("ollama")
class OllamaProvider(LLMProvider):
    """Ollama provider for local models."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from ollama import AsyncClient
        except ImportError:
            raise ImportError(
                "ollama package not installed. "
                "Install with: pip install modular-agents[ollama]"
            )
        
        self.client = AsyncClient(
            host=config.base_url or "http://localhost:11434"
        )
    
    @property
    def name(self) -> str:
        return "ollama"
    
    async def complete(
        self,
        messages: list[LLMMessage],
        system: str | None = None,
    ) -> LLMResponse:
        """Generate a completion using Ollama."""
        formatted = self.format_messages(messages, system)
        
        response = await self.client.chat(
            model=self.config.model,
            messages=formatted,
            options={
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        )
        
        return LLMResponse(
            content=response["message"]["content"],
            model=self.config.model,
            usage={
                "input_tokens": response.get("prompt_eval_count", 0),
                "output_tokens": response.get("eval_count", 0),
            },
            raw=response,
        )
    
    async def stream(
        self,
        messages: list[LLMMessage],
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion using Ollama."""
        formatted = self.format_messages(messages, system)

        stream = await self.client.chat(
            model=self.config.model,
            messages=formatted,
            options={
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
            stream=True,
        )

        async for chunk in stream:
            if chunk.get("message", {}).get("content"):
                yield chunk["message"]["content"]

    async def chat_with_tools(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
        max_tool_rounds: int = 5,
    ) -> tuple[str | None, list[ToolCall]]:
        """Chat with tool calling support.

        NOTE: Tool calling support varies by Ollama model.
        Some models may not support tool calling at all.
        This is a basic stub that falls back to regular completion.

        Args:
            messages: Conversation history
            system: System prompt
            tools: Available tools
            max_tool_rounds: Maximum tool calling iterations

        Returns:
            Tuple of (final_response, tool_calls_made)

        Raises:
            NotImplementedError: If tool calling is not supported by the model
        """
        # Ollama's tool calling support is model-dependent
        # For now, raise an error directing users to use Claude or OpenAI for tool calling
        raise NotImplementedError(
            "Tool calling is not yet implemented for Ollama provider. "
            "Some Ollama models may support tool calling, but it requires "
            "model-specific implementation. "
            "For tool calling support, use --provider claude or --provider openai. "
            "To disable tools with Ollama, use --no-tools flag."
        )
