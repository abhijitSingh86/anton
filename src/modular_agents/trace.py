"""Tracing and logging infrastructure for agent interactions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


@dataclass
class LLMInteraction:
    """Record of a single LLM interaction."""

    timestamp: datetime
    agent_name: str
    provider: str
    model: str
    system_prompt: str
    messages: list[dict[str, str]]
    response: str
    duration_ms: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "provider": self.provider,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "messages": self.messages,
            "response": self.response,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "metadata": self.metadata,
        }


class TraceLogger:
    """Centralized trace logging for all agent interactions.

    Features:
    - Logs all LLM calls with full context
    - Optionally displays in real-time
    - Saves traces to disk for post-mortem analysis
    - Supports different verbosity levels
    """

    def __init__(
        self,
        trace_dir: Path | None = None,
        verbose: bool = False,
        debug: bool = False,
    ):
        self.trace_dir = trace_dir
        self.verbose = verbose
        self.debug = debug
        self.interactions: list[LLMInteraction] = []

        # Set up Python logging
        log_level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("modular_agents")

        if trace_dir:
            trace_dir.mkdir(parents=True, exist_ok=True)

    def log_interaction(self, interaction: LLMInteraction) -> None:
        """Log an LLM interaction."""
        self.interactions.append(interaction)

        # Log to Python logger
        if interaction.error:
            self.logger.error(
                f"[{interaction.agent_name}] LLM call failed: {interaction.error}"
            )
        else:
            self.logger.info(
                f"[{interaction.agent_name}] LLM call completed in {interaction.duration_ms:.0f}ms"
            )

        # Display in real-time if verbose
        if self.verbose and not interaction.error:
            self._display_interaction(interaction)
        elif self.debug:
            self._display_interaction(interaction, full=True)

        # Save to disk
        if self.trace_dir:
            self._save_interaction(interaction)

    def _display_interaction(self, interaction: LLMInteraction, full: bool = False) -> None:
        """Display interaction in the console."""
        console.print(f"\n[dim]{'='*80}[/dim]")

        # Highlight module agents differently
        agent_display = interaction.agent_name
        if interaction.agent_name.startswith("module_"):
            module_name = interaction.agent_name.replace("module_", "")
            agent_display = f"🤖 Agent[{module_name}]"
        elif interaction.agent_name == "orchestrator":
            agent_display = "🎯 Orchestrator"

        console.print(
            f"[bold cyan]{agent_display}[/bold cyan] → "
            f"[dim]{interaction.provider}/{interaction.model}[/dim]"
        )
        console.print(f"[dim]{interaction.timestamp.strftime('%H:%M:%S')}[/dim]")

        if full or self.debug:
            # Show system prompt
            console.print("\n[bold]System Prompt:[/bold]")
            console.print(Panel(
                interaction.system_prompt[:500] + "..." if len(interaction.system_prompt) > 500 else interaction.system_prompt,
                border_style="dim",
            ))

            # Show messages
            console.print("\n[bold]Messages:[/bold]")
            for msg in interaction.messages:
                role_color = "green" if msg["role"] == "user" else "blue"
                content = msg["content"]
                if len(content) > 300 and not self.debug:
                    content = content[:300] + "..."
                console.print(f"[{role_color}]{msg['role']}:[/{role_color}] {content}")

        # Show response (always show in verbose mode)
        console.print("\n[bold]Response:[/bold]")
        response_text = interaction.response
        if len(response_text) > 500 and not self.debug:
            response_text = response_text[:500] + "..."

        # Try to highlight JSON in responses
        if "{" in response_text and "}" in response_text:
            try:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_part = response_text[start:end]
                parsed = json.loads(json_part)
                syntax = Syntax(
                    json.dumps(parsed, indent=2),
                    "json",
                    theme="monokai",
                    line_numbers=False,
                )
                console.print(syntax)
            except (json.JSONDecodeError, ValueError):
                console.print(response_text)
        else:
            console.print(response_text)

        if interaction.duration_ms:
            console.print(f"\n[dim]Duration: {interaction.duration_ms:.0f}ms[/dim]")

        if interaction.error:
            console.print(f"\n[red]Error: {interaction.error}[/red]")

    def _save_interaction(self, interaction: LLMInteraction) -> None:
        """Save interaction to disk."""
        if not self.trace_dir:
            return

        # Create a file per agent per session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{interaction.agent_name}_{timestamp}.jsonl"
        filepath = self.trace_dir / filename

        # Append as JSONL (one JSON object per line)
        with open(filepath, "a") as f:
            f.write(json.dumps(interaction.to_dict()) + "\n")

    def save_summary(self, filename: str = "trace_summary.json") -> None:
        """Save a summary of all interactions."""
        if not self.trace_dir:
            return

        summary = {
            "total_interactions": len(self.interactions),
            "by_agent": {},
            "errors": [],
            "interactions": [i.to_dict() for i in self.interactions],
        }

        for interaction in self.interactions:
            agent = interaction.agent_name
            if agent not in summary["by_agent"]:
                summary["by_agent"][agent] = {"count": 0, "errors": 0}
            summary["by_agent"][agent]["count"] += 1
            if interaction.error:
                summary["by_agent"][agent]["errors"] += 1
                summary["errors"].append({
                    "agent": agent,
                    "error": interaction.error,
                    "timestamp": interaction.timestamp.isoformat(),
                })

        filepath = self.trace_dir / filename
        filepath.write_text(json.dumps(summary, indent=2))
        console.print(f"\n[green]Trace summary saved to {filepath}[/green]")


# Global trace logger instance
_trace_logger: TraceLogger | None = None


def init_trace_logger(
    trace_dir: Path | None = None,
    verbose: bool = False,
    debug: bool = False,
) -> TraceLogger:
    """Initialize the global trace logger."""
    global _trace_logger
    _trace_logger = TraceLogger(trace_dir=trace_dir, verbose=verbose, debug=debug)
    return _trace_logger


def get_trace_logger() -> TraceLogger | None:
    """Get the global trace logger instance."""
    return _trace_logger


def log_llm_interaction(interaction: LLMInteraction) -> None:
    """Log an LLM interaction to the global trace logger."""
    if _trace_logger:
        _trace_logger.log_interaction(interaction)
