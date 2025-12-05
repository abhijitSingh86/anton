"""Base agent class."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

from modular_agents.llm import LLMMessage, LLMProvider

if TYPE_CHECKING:
    from modular_agents.core.models import AgentMessage


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(
        self,
        name: str,
        llm: LLMProvider,
        system_prompt: str = "",
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.conversation_history: list[LLMMessage] = []
    
    @abstractmethod
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process an incoming message and return a response."""
        ...
    
    async def think(self, user_message: str) -> str:
        """Send a message to the LLM and get a response."""
        self.conversation_history.append(
            LLMMessage(role="user", content=user_message)
        )

        response = await self.llm.complete(
            messages=self.conversation_history,
            system=self.system_prompt,
        )

        self.conversation_history.append(
            LLMMessage(role="assistant", content=response.content)
        )

        return response.content
    
    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def add_context(self, context: str):
        """Add context to the conversation without expecting a response."""
        self.conversation_history.append(
            LLMMessage(role="user", content=f"[Context]\n{context}")
        )
        self.conversation_history.append(
            LLMMessage(role="assistant", content="Understood. I've noted this context.")
        )
