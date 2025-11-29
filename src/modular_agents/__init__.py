"""Modular Agents - Multi-agent framework for modular codebases.

A pluggable framework that:
1. Analyzes modular codebases (SBT, Maven, npm, etc.)
2. Creates specialized agents for each module
3. Coordinates task execution across modules
4. Supports multiple LLM backends (Claude, OpenAI, Ollama)

Quick Start:
    # Install with Claude support
    pip install modular-agents[claude]
    
    # Run interactive mode
    modular-agents run /path/to/repo --provider claude
    
    # Or use programmatically
    from modular_agents import AgentRuntime, LLMConfig
    
    config = LLMConfig(model="claude-sonnet-4-20250514", api_key="...")
    runtime = AgentRuntime("/path/to/repo", llm_config=config)
    await runtime.initialize()
    result = await runtime.execute_task("Add logging to UserService")
"""

__version__ = "0.1.0"

from modular_agents.core import (
    ModuleProfile,
    ProjectType,
    RepoKnowledge,
    Task,
    TaskResult,
    TaskStatus,
)
from modular_agents.llm import LLMConfig, LLMProvider, LLMProviderRegistry
from modular_agents.runtime import AgentRuntime, InteractiveLoop

__all__ = [
    # Runtime
    "AgentRuntime",
    "InteractiveLoop",
    # LLM
    "LLMConfig",
    "LLMProvider",
    "LLMProviderRegistry",
    # Models
    "ModuleProfile",
    "ProjectType",
    "RepoKnowledge",
    "Task",
    "TaskResult",
    "TaskStatus",
]
