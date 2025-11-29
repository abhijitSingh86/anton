"""Checkpoint management for task execution continuation."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from modular_agents.core.models import (
    ExecutionPlan,
    SubTask,
    SubTaskResult,
    TaskStatus,
)


# =============================================================================
# Agent State Models
# =============================================================================


@dataclass
class LLMMessage:
    """A message in the conversation history."""

    role: str  # "system", "user", "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMMessage:
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class AgentState:
    """State of a single agent at checkpoint time."""

    agent_name: str
    conversation_history: list[LLMMessage] = field(default_factory=list)
    retry_count: dict[str, int] = field(default_factory=dict)  # subtask_id -> count
    last_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "agent_name": self.agent_name,
            "conversation_history": [msg.to_dict() for msg in self.conversation_history],
            "retry_count": self.retry_count,
            "last_error": self.last_error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentState:
        """Create from dictionary."""
        return cls(
            agent_name=data["agent_name"],
            conversation_history=[
                LLMMessage.from_dict(msg) for msg in data.get("conversation_history", [])
            ],
            retry_count=data.get("retry_count", {}),
            last_error=data.get("last_error"),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Checkpoint Model
# =============================================================================


@dataclass
class Checkpoint:
    """Represents a task execution checkpoint."""

    task_id: str
    checkpoint_id: str
    timestamp: datetime
    phase_index: int
    total_phases: int
    completed_subtasks: list[SubTaskResult]
    pending_subtasks: list[SubTask]
    execution_plan: ExecutionPlan
    agent_states: dict[str, AgentState]
    task_description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "checkpoint_id": self.checkpoint_id,
            "timestamp": self.timestamp.isoformat(),
            "phase_index": self.phase_index,
            "total_phases": self.total_phases,
            "completed_subtasks": [
                result.model_dump() for result in self.completed_subtasks
            ],
            "pending_subtasks": [task.model_dump() for task in self.pending_subtasks],
            "execution_plan": self.execution_plan.model_dump(),
            "agent_states": {
                name: state.to_dict() for name, state in self.agent_states.items()
            },
            "task_description": self.task_description,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        """Create from dictionary."""
        return cls(
            task_id=data["task_id"],
            checkpoint_id=data["checkpoint_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            phase_index=data["phase_index"],
            total_phases=data["total_phases"],
            completed_subtasks=[
                SubTaskResult(**result) for result in data["completed_subtasks"]
            ],
            pending_subtasks=[SubTask(**task) for task in data["pending_subtasks"]],
            execution_plan=ExecutionPlan(**data["execution_plan"]),
            agent_states={
                name: AgentState.from_dict(state)
                for name, state in data["agent_states"].items()
            },
            task_description=data.get("task_description", ""),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Checkpoint Manager
# =============================================================================


class CheckpointManager:
    """Manages task checkpoints and resumption."""

    def __init__(self, base_path: Path | None = None):
        """
        Initialize checkpoint manager.

        Args:
            base_path: Base directory for checkpoints. Defaults to .modular-agents/checkpoints
        """
        self.base_path = (
            base_path or Path.cwd() / ".modular-agents" / "checkpoints"
        )
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_task_dir(self, task_id: str) -> Path:
        """Get directory for a specific task's checkpoints."""
        task_dir = self.base_path / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir

    def _get_checkpoint_file(self, task_id: str, checkpoint_id: str) -> Path:
        """Get file path for a specific checkpoint."""
        return self._get_task_dir(task_id) / f"checkpoint_{checkpoint_id}.json"

    def _get_metadata_file(self, task_id: str) -> Path:
        """Get metadata file path for a task."""
        return self._get_task_dir(task_id) / "metadata.json"

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """
        Save checkpoint to disk.

        Args:
            checkpoint: Checkpoint to save
        """
        # Save checkpoint file
        checkpoint_file = self._get_checkpoint_file(
            checkpoint.task_id, checkpoint.checkpoint_id
        )
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

        # Update metadata
        self._update_metadata(checkpoint)

    def _update_metadata(self, checkpoint: Checkpoint) -> None:
        """Update task metadata with latest checkpoint info."""
        metadata_file = self._get_metadata_file(checkpoint.task_id)

        # Load existing metadata or create new
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
        else:
            metadata = {
                "task_id": checkpoint.task_id,
                "task_description": checkpoint.task_description,
                "created_at": datetime.now().isoformat(),
                "checkpoints": [],
            }

        # Add checkpoint info
        checkpoint_info = {
            "checkpoint_id": checkpoint.checkpoint_id,
            "timestamp": checkpoint.timestamp.isoformat(),
            "phase_index": checkpoint.phase_index,
            "total_phases": checkpoint.total_phases,
            "completed_subtasks": len(checkpoint.completed_subtasks),
            "pending_subtasks": len(checkpoint.pending_subtasks),
        }

        # Update or append
        existing = [
            c for c in metadata["checkpoints"]
            if c["checkpoint_id"] != checkpoint.checkpoint_id
        ]
        existing.append(checkpoint_info)
        metadata["checkpoints"] = sorted(
            existing, key=lambda c: c["timestamp"]
        )
        metadata["latest_checkpoint"] = checkpoint.checkpoint_id
        metadata["updated_at"] = datetime.now().isoformat()

        # Save metadata
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def load_checkpoint(
        self, task_id: str, checkpoint_id: str | None = None
    ) -> Checkpoint | None:
        """
        Load checkpoint from disk.

        Args:
            task_id: Task ID
            checkpoint_id: Specific checkpoint ID. If None, loads latest.

        Returns:
            Checkpoint if found, None otherwise
        """
        task_dir = self._get_task_dir(task_id)
        if not task_dir.exists():
            return None

        # Get checkpoint ID
        if checkpoint_id is None:
            # Load latest from metadata
            metadata_file = self._get_metadata_file(task_id)
            if not metadata_file.exists():
                return None

            with open(metadata_file) as f:
                metadata = json.load(f)

            checkpoint_id = metadata.get("latest_checkpoint")
            if not checkpoint_id:
                return None

        # Load checkpoint
        checkpoint_file = self._get_checkpoint_file(task_id, checkpoint_id)
        if not checkpoint_file.exists():
            return None

        with open(checkpoint_file) as f:
            data = json.load(f)

        return Checkpoint.from_dict(data)

    def list_checkpoints(self, task_id: str) -> list[dict[str, Any]]:
        """
        List all checkpoints for a task.

        Args:
            task_id: Task ID

        Returns:
            List of checkpoint info dictionaries
        """
        metadata_file = self._get_metadata_file(task_id)
        if not metadata_file.exists():
            return []

        with open(metadata_file) as f:
            metadata = json.load(f)

        return metadata.get("checkpoints", [])

    def list_all_tasks(self) -> list[dict[str, Any]]:
        """
        List all tasks with checkpoints.

        Returns:
            List of task metadata
        """
        tasks = []
        for task_dir in self.base_path.iterdir():
            if not task_dir.is_dir():
                continue

            metadata_file = task_dir / "metadata.json"
            if not metadata_file.exists():
                continue

            with open(metadata_file) as f:
                metadata = json.load(f)
                tasks.append(metadata)

        return sorted(tasks, key=lambda t: t.get("created_at", ""), reverse=True)

    def delete_checkpoint(self, task_id: str, checkpoint_id: str) -> bool:
        """
        Delete a specific checkpoint.

        Args:
            task_id: Task ID
            checkpoint_id: Checkpoint ID

        Returns:
            True if deleted, False if not found
        """
        checkpoint_file = self._get_checkpoint_file(task_id, checkpoint_id)
        if not checkpoint_file.exists():
            return False

        checkpoint_file.unlink()

        # Update metadata
        metadata_file = self._get_metadata_file(task_id)
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)

            # Remove from checkpoints list
            metadata["checkpoints"] = [
                c for c in metadata.get("checkpoints", [])
                if c["checkpoint_id"] != checkpoint_id
            ]

            # Update latest if needed
            if metadata.get("latest_checkpoint") == checkpoint_id:
                if metadata["checkpoints"]:
                    metadata["latest_checkpoint"] = metadata["checkpoints"][-1][
                        "checkpoint_id"
                    ]
                else:
                    metadata["latest_checkpoint"] = None

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

        return True

    def delete_task_checkpoints(self, task_id: str) -> bool:
        """
        Delete all checkpoints for a task.

        Args:
            task_id: Task ID

        Returns:
            True if deleted, False if not found
        """
        task_dir = self._get_task_dir(task_id)
        if not task_dir.exists():
            return False

        # Remove all checkpoint files
        for checkpoint_file in task_dir.glob("checkpoint_*.json"):
            checkpoint_file.unlink()

        # Remove metadata
        metadata_file = self._get_metadata_file(task_id)
        if metadata_file.exists():
            metadata_file.unlink()

        # Remove directory if empty
        try:
            task_dir.rmdir()
        except OSError:
            pass  # Directory not empty

        return True

    def cleanup_old_checkpoints(self, days: int = 7) -> int:
        """
        Remove checkpoints older than N days.

        Args:
            days: Age threshold in days

        Returns:
            Number of checkpoints removed
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=days)
        removed = 0

        for task_dir in self.base_path.iterdir():
            if not task_dir.is_dir():
                continue

            metadata_file = task_dir / "metadata.json"
            if not metadata_file.exists():
                continue

            with open(metadata_file) as f:
                metadata = json.load(f)

            # Check each checkpoint
            kept_checkpoints = []
            for checkpoint_info in metadata.get("checkpoints", []):
                timestamp = datetime.fromisoformat(checkpoint_info["timestamp"])
                if timestamp < cutoff:
                    # Remove checkpoint file
                    checkpoint_id = checkpoint_info["checkpoint_id"]
                    checkpoint_file = self._get_checkpoint_file(
                        metadata["task_id"], checkpoint_id
                    )
                    if checkpoint_file.exists():
                        checkpoint_file.unlink()
                        removed += 1
                else:
                    kept_checkpoints.append(checkpoint_info)

            # Update metadata
            if len(kept_checkpoints) < len(metadata.get("checkpoints", [])):
                metadata["checkpoints"] = kept_checkpoints
                if kept_checkpoints:
                    metadata["latest_checkpoint"] = kept_checkpoints[-1][
                        "checkpoint_id"
                    ]
                else:
                    metadata["latest_checkpoint"] = None

                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)

            # Remove empty task directories
            if not kept_checkpoints:
                try:
                    metadata_file.unlink()
                    task_dir.rmdir()
                except OSError:
                    pass

        return removed

    def create_checkpoint(
        self,
        task_id: str,
        task_description: str,
        phase_index: int,
        total_phases: int,
        completed_subtasks: list[SubTaskResult],
        pending_subtasks: list[SubTask],
        execution_plan: ExecutionPlan,
        agent_states: dict[str, AgentState],
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """
        Create a new checkpoint.

        Args:
            task_id: Task ID
            task_description: Description of the task
            phase_index: Current phase index
            total_phases: Total number of phases
            completed_subtasks: List of completed subtask results
            pending_subtasks: List of pending subtasks
            execution_plan: Execution plan
            agent_states: Dictionary of agent states
            metadata: Optional metadata

        Returns:
            Created checkpoint
        """
        checkpoint = Checkpoint(
            task_id=task_id,
            checkpoint_id=f"phase{phase_index}_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(),
            phase_index=phase_index,
            total_phases=total_phases,
            completed_subtasks=completed_subtasks,
            pending_subtasks=pending_subtasks,
            execution_plan=execution_plan,
            agent_states=agent_states,
            task_description=task_description,
            metadata=metadata or {},
        )

        self.save_checkpoint(checkpoint)
        return checkpoint
