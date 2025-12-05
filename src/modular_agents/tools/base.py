"""Base classes for tool calling system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""

    id: str
    name: str
    parameters: dict[str, Any]


@dataclass
class ToolResult:
    """Result from executing a tool."""

    tool_call_id: str
    tool_name: str
    success: bool
    output: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for LLM response."""
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
        }


class Tool(ABC):
    """Base class for all tools.

    Each tool must:
    1. Define its name and description
    2. Define input parameters (JSON schema)
    3. Implement execute() method
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (used by LLM to invoke)."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does."""
        pass

    @property
    @abstractmethod
    def parameters_schema(self) -> dict:
        """JSON schema for tool parameters."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters.

        Args:
            **kwargs: Parameters defined in parameters_schema

        Returns:
            ToolResult with execution outcome
        """
        pass

    def to_anthropic_format(self) -> dict:
        """Convert tool definition to Anthropic API format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters_schema,
        }

    def to_openai_format(self) -> dict:
        """Convert tool definition to OpenAI API format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool.

        Args:
            tool: Tool instance to register
        """
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools.

        Returns:
            List of all tools
        """
        return list(self._tools.values())

    def to_anthropic_format(self) -> list[dict]:
        """Convert all tools to Anthropic API format.

        Returns:
            List of tool definitions for Anthropic
        """
        return [tool.to_anthropic_format() for tool in self._tools.values()]

    def to_openai_format(self) -> list[dict]:
        """Convert all tools to OpenAI API format.

        Returns:
            List of tool definitions for OpenAI
        """
        return [tool.to_openai_format() for tool in self._tools.values()]

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call.

        Args:
            tool_call: ToolCall to execute

        Returns:
            ToolResult with execution outcome
        """
        tool = self.get(tool_call.name)

        if not tool:
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=False,
                output="",
                error=f"Tool '{tool_call.name}' not found",
            )

        try:
            result = await tool.execute(**tool_call.parameters)
            result.tool_call_id = tool_call.id
            return result
        except Exception as e:
            error_msg = f"Tool execution failed: {type(e).__name__}: {str(e)}"
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=False,
                output="",
                error=error_msg,
            )
