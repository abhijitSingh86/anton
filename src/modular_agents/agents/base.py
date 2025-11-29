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
        from modular_agents.trace import LLMInteraction, log_llm_interaction

        self.conversation_history.append(
            LLMMessage(role="user", content=user_message)
        )

        # Track timing and log interaction
        start_time = time.time()
        error = None
        response_content = ""

        try:
            response = await self.llm.complete(
                messages=self.conversation_history,
                system=self.system_prompt,
            )
            response_content = response.content

            self.conversation_history.append(
                LLMMessage(role="assistant", content=response.content)
            )
        except Exception as e:
            error = str(e)
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000

            # Log the interaction
            interaction = LLMInteraction(
                timestamp=datetime.now(),
                agent_name=self.name,
                provider=getattr(self.llm, "__class__.__name__", "unknown"),
                model=getattr(self.llm, "config", None) and self.llm.config.model or "unknown",
                system_prompt=self.system_prompt,
                messages=[{"role": m.role, "content": m.content} for m in self.conversation_history],
                response=response_content,
                duration_ms=duration_ms,
                error=error,
            )
            log_llm_interaction(interaction)

        return response_content
    
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
