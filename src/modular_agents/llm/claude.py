"""Claude (Anthropic) LLM provider."""

from __future__ import annotations

from typing import AsyncIterator

from .base import LLMConfig, LLMMessage, LLMProvider, LLMProviderRegistry, LLMResponse
from modular_agents.tools.base import ToolCall


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
            default_headers=config.extra.get("headers", {}),
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

    async def chat_with_tools(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
        max_tool_rounds: int = 5,
    ) -> tuple[str | None, list[ToolCall]]:
        """Chat with tool calling support using Claude's tools API.

        Args:
            messages: Conversation history in Anthropic format
            system: System prompt
            tools: Available tools in Anthropic format
            max_tool_rounds: Maximum tool calling iterations

        Returns:
            Tuple of (final_response, tool_calls_made)
        """
        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": system,
            "messages": messages,
            "tools": tools,
        }

        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature

        # Call Claude API
        response = await self.client.messages.create(**kwargs)

        # Check if Claude wants to use tools
        if response.stop_reason == "tool_use":
            # Extract tool calls from response
            tool_calls = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            parameters=block.input,
                        )
                    )
            return None, tool_calls
        else:
            # Regular text response
            text_content = ""
            for block in response.content:
                if block.type == "text":
                    text_content += block.text
            return text_content, []
