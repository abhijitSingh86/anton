"""LLM providers package.

This package provides a pluggable abstraction layer for different LLM providers.
Supported providers:
- claude: Anthropic Claude (requires `pip install modular-agents[claude]`)
- openai: OpenAI and compatible APIs (requires `pip install modular-agents[openai]`)
- ollama: Local Ollama models (requires `pip install modular-agents[ollama]`)

Usage:
    from modular_agents.llm import LLMProviderRegistry, LLMConfig
    
    config = LLMConfig(model="claude-sonnet-4-20250514", api_key="...")
    provider = LLMProviderRegistry.create("claude", config)
    
    response = await provider.complete([
        LLMMessage(role="user", content="Hello!")
    ])
"""

from .base import (
    LLMConfig,
    LLMMessage,
    LLMProvider,
    LLMProviderRegistry,
    LLMResponse,
)

# Auto-register providers (they self-register via decorator)
# We import them here so they register when the package is imported
_PROVIDER_MODULES = ["claude", "openai", "ollama"]

for _module in _PROVIDER_MODULES:
    try:
        __import__(f"modular_agents.llm.{_module}")
    except ImportError:
        # Provider's dependencies not installed, skip
        pass


__all__ = [
    "LLMConfig",
    "LLMMessage",
    "LLMProvider",
    "LLMProviderRegistry",
    "LLMResponse",
]
