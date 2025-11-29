"""Ollama provider for local LLMs."""

from __future__ import annotations

from typing import AsyncIterator

from .base import LLMConfig, LLMMessage, LLMProvider, LLMProviderRegistry, LLMResponse


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
