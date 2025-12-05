"""Tool calling system for agents."""

from .base import Tool, ToolCall, ToolRegistry, ToolResult
from .code_tools import GrepCodebaseTool, ReadFileTool, SearchKnowledgeTool
from .execution_tools import RunCommandTool, RunTestsTool, ValidateSyntaxTool
from .git_tools import GetDependenciesTool, GetGitHistoryTool

__all__ = [
    # Base
    "Tool",
    "ToolCall",
    "ToolResult",
    "ToolRegistry",
    # Code Tools
    "ReadFileTool",
    "GrepCodebaseTool",
    "SearchKnowledgeTool",
    # Execution Tools
    "RunCommandTool",
    "RunTestsTool",
    "ValidateSyntaxTool",
    # Analysis Tools
    "GetDependenciesTool",
    "GetGitHistoryTool",
]


def register_default_tools(
    repo_path: str,
    knowledge_store=None,
    allow_commands: bool = True,
) -> ToolRegistry:
    """Register all default tools.

    Args:
        repo_path: Repository root path
        knowledge_store: Optional knowledge store for search
        allow_commands: Whether to allow command execution (default: True)

    Returns:
        ToolRegistry with all tools registered
    """
    registry = ToolRegistry()

    # Code tools (always safe)
    registry.register(ReadFileTool(repo_path))
    registry.register(GrepCodebaseTool(repo_path))

    if knowledge_store:
        registry.register(SearchKnowledgeTool(knowledge_store, repo_path))

    # Analysis tools (safe, read-only)
    registry.register(GetDependenciesTool(repo_path))
    registry.register(GetGitHistoryTool(repo_path))

    # Execution tools (potentially unsafe, gated by allow_commands)
    if allow_commands:
        registry.register(RunCommandTool(repo_path))
        registry.register(RunTestsTool(repo_path))
        registry.register(ValidateSyntaxTool(repo_path))

    return registry
