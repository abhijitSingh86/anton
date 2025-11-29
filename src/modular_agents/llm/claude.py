"""Claude (Anthropic) LLM provider."""

from __future__ import annotations

from typing import AsyncIterator

from .base import LLMConfig, LLMMessage, LLMProvider, LLMProviderRegistry, LLMResponse


@LLMProviderRegistry.register("claude")
class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install modular-agents[claude]"
            )
        
        self.client = AsyncAnthropic(
            api_key=config.api_key,
            base_url=config.base_url,
        )
    
    @property
    def name(self) -> str:
        return "claude"
    
    async def complete(
        self,
        messages: list[LLMMessage],
        system: str | None = None,
    ) -> LLMResponse:
        """Generate a completion using Claude."""
        formatted = [{"role": m.role, "content": m.content} for m in messages]
        
        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": formatted,
        }
        
        if system:
            kwargs["system"] = system
        
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        
        response = await self.client.messages.create(**kwargs)
        
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            raw=response.model_dump(),
        )
    
    async def stream(
        self,
        messages: list[LLMMessage],
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion using Claude."""
        formatted = [{"role": m.role, "content": m.content} for m in messages]
        
        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": formatted,
        }
        
        if system:
            kwargs["system"] = system
        
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        
        async with self.client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
