"""OpenAI-compatible LLM provider (works with OpenAI, Azure, local servers)."""

from __future__ import annotations

import json
from typing import AsyncIterator

from .base import LLMConfig, LLMMessage, LLMProvider, LLMProviderRegistry, LLMResponse
from modular_agents.tools.base import ToolCall


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
            default_headers=config.extra.get("headers", {}),
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

    async def chat_with_tools(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
        max_tool_rounds: int = 5,
    ) -> tuple[str | None, list[ToolCall]]:
        """Chat with tool calling support using OpenAI's function calling API.

        Args:
            messages: Conversation history in OpenAI format
            system: System prompt
            tools: Available tools in OpenAI format
            max_tool_rounds: Maximum tool calling iterations

        Returns:
            Tuple of (final_response, tool_calls_made)

        Note:
            If the server doesn't support tool calling, falls back to regular completion.
        """
        # Add system message to conversation
        messages_with_system = [
            {"role": "system", "content": system},
            *messages,
        ]

        try:
            # Try calling with tools first
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages_with_system,
                tools=tools,
                tool_choice="auto",
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            message = response.choices[0].message

            # Check if OpenAI wants to use tools
            if message.tool_calls:
                # Extract tool calls
                tool_calls = []
                for tc in message.tool_calls:
                    tool_calls.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            parameters=json.loads(tc.function.arguments),
                        )
                    )
                return None, tool_calls
            else:
                # Regular text response
                return message.content or "", []

        except Exception as e:
            # Check if it's a tool calling error
            error_str = str(e).lower()
            if any(indicator in error_str for indicator in [
                "tools param",
                "jinja",
                "function calling",
                "not supported",
                "invalid parameter",
            ]):
                # Server doesn't support tool calling - fallback to regular completion
                # Just return the last message as a regular completion
                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages_with_system,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                return response.choices[0].message.content or "", []
            else:
                # Some other error - re-raise
                raise
