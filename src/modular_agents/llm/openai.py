"""OpenAI-compatible LLM provider (works with OpenAI, Azure, local servers)."""

from __future__ import annotations

from typing import AsyncIterator

from .base import LLMConfig, LLMMessage, LLMProvider, LLMProviderRegistry, LLMResponse


@LLMProviderRegistry.register("openai")
class OpenAIProvider(LLMProvider):
    """OpenAI-compatible provider (works with any OpenAI-compatible API)."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install modular-agents[openai]"
            )
        
        self.client = AsyncOpenAI(
            api_key=config.api_key or "not-needed",  # Some local servers don't need keys
            base_url=config.base_url,
        )
    
    @property
    def name(self) -> str:
        return "openai"
    
    async def complete(
        self,
        messages: list[LLMMessage],
        system: str | None = None,
    ) -> LLMResponse:
        """Generate a completion."""
        formatted = self.format_messages(messages, system)
        
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=formatted,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        
        choice = response.choices[0]
        
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            raw=response.model_dump(),
        )
    
    async def stream(
        self,
        messages: list[LLMMessage],
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion."""
        formatted = self.format_messages(messages, system)
        
        stream = await self.client.chat.completions.create(
            model=self.config.model,
            messages=formatted,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
