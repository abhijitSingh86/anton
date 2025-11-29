"""Progress tracking and persistence for task execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from modular_agents.core.models import ExecutionPlan, SubTaskResult, TaskStatus


# =============================================================================
# Progress State
# =============================================================================


@dataclass
class ProgressState:
    """Current progress state of task execution."""

    task_id: str
    task_description: str
    status: TaskStatus
    total_phases: int
    completed_phases: int
    current_phase: int
    total_subtasks: int
    completed_subtasks: int
    failed_subtasks: int
    blocked_subtasks: int
    retry_attempts: dict[str, int] = field(default_factory=dict)  # subtask_id -> count
    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    estimated_completion: datetime | None = None
    checkpoint_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "status": self.status.value,
            "total_phases": self.total_phases,
            "completed_phases": self.completed_phases,
            "current_phase": self.current_phase,
            "total_subtasks": self.total_subtasks,
            "completed_subtasks": self.completed_subtasks,
            "failed_subtasks": self.failed_subtasks,
            "blocked_subtasks": self.blocked_subtasks,
            "retry_attempts": self.retry_attempts,
            "start_time": self.start_time.isoformat(),
            "last_update": self.last_update.isoformat(),
            "estimated_completion": (
                self.estimated_completion.isoformat()
                if self.estimated_completion
                else None
            ),
            "checkpoint_path": self.checkpoint_path,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProgressState:
        """Create from dictionary."""
        return cls(
            task_id=data["task_id"],
            task_description=data["task_description"],
            status=TaskStatus(data["status"]),
            total_phases=data["total_phases"],
            completed_phases=data["completed_phases"],
            current_phase=data["current_phase"],
            total_subtasks=data["total_subtasks"],
            completed_subtasks=data["completed_subtasks"],
            failed_subtasks=data["failed_subtasks"],
            blocked_subtasks=data["blocked_subtasks"],
            retry_attempts=data.get("retry_attempts", {}),
            start_time=datetime.fromisoformat(data["start_time"]),
            last_update=datetime.fromisoformat(data["last_update"]),
            estimated_completion=(
                datetime.fromisoformat(data["estimated_completion"])
                if data.get("estimated_completion")
                else None
            ),
            checkpoint_path=data.get("checkpoint_path"),
            metadata=data.get("metadata", {}),
        )

    @property
    def elapsed_time(self) -> timedelta:
        """Get elapsed time since start."""
        return datetime.now() - self.start_time

    @property
    def progress_percentage(self) -> float:
        """Get overall progress percentage."""
        if self.total_subtasks == 0:
            return 0.0
        return (self.completed_subtasks / self.total_subtasks) * 100

    @property
    def phase_percentage(self) -> float:
        """Get phase progress percentage."""
        if self.total_phases == 0:
            return 0.0
        return (self.completed_phases / self.total_phases) * 100


# =============================================================================
# Progress Tracker
# =============================================================================


class ProgressTracker:
    """Tracks and persists task execution progress."""

    def __init__(self, progress_dir: Path | None = None):
        """
        Initialize progress tracker.

        Args:
            progress_dir: Directory for progress files (defaults to .modular-agents/progress)
        """
        self.progress_dir = (
            progress_dir or Path.cwd() / ".modular-agents" / "progress"
        )
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console()
        self.current_state: ProgressState | None = None
        self.rich_progress: Progress | None = None
        self.rich_task_id: TaskID | None = None

    def _get_progress_file(self, task_id: str) -> Path:
        """Get progress file path for a task."""
        return self.progress_dir / f"{task_id}.json"

    def start_task(
        self,
        task_id: str,
        task_description: str,
        total_subtasks: int = 0,
        total_phases: int = 0,
    ) -> None:
        """
        Start tracking a new task.

        Args:
            task_id: Task ID
            task_description: Task description
            total_subtasks: Total number of subtasks
            total_phases: Total number of phases
        """
        self.current_state = ProgressState(
            task_id=task_id,
            task_description=task_description,
            status=TaskStatus.IN_PROGRESS,
            total_phases=total_phases,
            completed_phases=0,
            current_phase=0,
            total_subtasks=total_subtasks,
            completed_subtasks=0,
            failed_subtasks=0,
            blocked_subtasks=0,
        )
        self.save_progress()

    def set_plan(self, execution_plan: ExecutionPlan) -> None:
        """
        Set execution plan to update total counts.

        Args:
            execution_plan: Execution plan
        """
        if not self.current_state:
            return

        self.current_state.total_phases = len(execution_plan.phases)
        total_subtasks = sum(len(phase.subtask_ids) for phase in execution_plan.phases)
        self.current_state.total_subtasks = total_subtasks

        # Estimate completion time based on average subtask time
        # Assume 2 minutes per subtask as initial estimate
        estimated_minutes = total_subtasks * 2
        self.current_state.estimated_completion = datetime.now() + timedelta(
            minutes=estimated_minutes
        )

        self.save_progress()

    def phase_started(self, phase_index: int) -> None:
        """
        Mark phase as started.

        Args:
            phase_index: Phase index
        """
        if not self.current_state:
            return

        self.current_state.current_phase = phase_index
        self.current_state.last_update = datetime.now()
        self.save_progress()

    def phase_completed(self, phase_index: int) -> None:
        """
        Mark phase as completed.

        Args:
            phase_index: Phase index
        """
        if not self.current_state:
            return

        self.current_state.completed_phases += 1
        self.current_state.last_update = datetime.now()

        # Update completion estimate
        self._update_estimate()

        self.save_progress()

    def subtask_started(self, subtask_id: str) -> None:
        """
        Mark subtask as started.

        Args:
            subtask_id: Subtask ID
        """
        if not self.current_state:
            return

        self.current_state.last_update = datetime.now()
        self.save_progress()

    def subtask_completed(self, subtask_id: str, result: SubTaskResult) -> None:
        """
        Mark subtask as completed.

        Args:
            subtask_id: Subtask ID
            result: Subtask result
        """
        if not self.current_state:
            return

        if result.status == TaskStatus.COMPLETED:
            self.current_state.completed_subtasks += 1
        elif result.status == TaskStatus.FAILED:
            self.current_state.failed_subtasks += 1
        elif result.status == TaskStatus.BLOCKED:
            self.current_state.blocked_subtasks += 1

        self.current_state.last_update = datetime.now()
        self._update_estimate()
        self.save_progress()

    def subtask_retry(self, subtask_id: str, attempt: int) -> None:
        """
        Record subtask retry attempt.

        Args:
            subtask_id: Subtask ID
            attempt: Attempt number
        """
        if not self.current_state:
            return

        self.current_state.retry_attempts[subtask_id] = attempt
        self.current_state.last_update = datetime.now()
        self.save_progress()

    def task_completed(self, status: TaskStatus = TaskStatus.COMPLETED) -> None:
        """
        Mark task as completed.

        Args:
            status: Final task status
        """
        if not self.current_state:
            return

        self.current_state.status = status
        self.current_state.last_update = datetime.now()
        self.save_progress()

    def update_checkpoint(self, checkpoint_path: str) -> None:
        """
        Update checkpoint path.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not self.current_state:
            return

        self.current_state.checkpoint_path = checkpoint_path
        self.save_progress()

    def _update_estimate(self) -> None:
        """Update estimated completion time based on current progress."""
        if not self.current_state:
            return

        if self.current_state.completed_subtasks == 0:
            return

        # Calculate average time per subtask
        elapsed = self.current_state.elapsed_time.total_seconds()
        avg_time = elapsed / self.current_state.completed_subtasks

        # Estimate remaining time
        remaining_tasks = (
            self.current_state.total_subtasks
            - self.current_state.completed_subtasks
        )
        estimated_remaining_seconds = avg_time * remaining_tasks

        self.current_state.estimated_completion = datetime.now() + timedelta(
            seconds=estimated_remaining_seconds
        )

    def get_current_state(self) -> ProgressState | None:
        """
        Get current progress state.

        Returns:
            Current progress state or None
        """
        return self.current_state

    def save_progress(self) -> None:
        """Persist progress to disk."""
        if not self.current_state:
            return

        progress_file = self._get_progress_file(self.current_state.task_id)
        with open(progress_file, "w") as f:
            json.dump(self.current_state.to_dict(), f, indent=2)

    def load_progress(self, task_id: str) -> ProgressState | None:
        """
        Load saved progress.

        Args:
            task_id: Task ID

        Returns:
            Progress state if found, None otherwise
        """
        progress_file = self._get_progress_file(task_id)
        if not progress_file.exists():
            return None

        with open(progress_file) as f:
            data = json.load(f)

        state = ProgressState.from_dict(data)
        self.current_state = state
        return state

    def display_progress(self, live: bool = False) -> None:
        """
        Show progress dashboard.

        Args:
            live: Whether to show live updating dashboard
        """
        if not self.current_state:
            self.console.print("[yellow]No active task[/yellow]")
            return

        if live:
            with Live(self._generate_progress_table(), console=self.console, refresh_per_second=1) as live_display:
                # Keep updating until task is complete
                import time

                while self.current_state and self.current_state.status == TaskStatus.IN_PROGRESS:
                    time.sleep(1)
                    # Reload state from disk
                    self.load_progress(self.current_state.task_id)
                    live_display.update(self._generate_progress_table())
        else:
            self.console.print(self._generate_progress_table())

    def _generate_progress_table(self) -> Table:
        """Generate progress table for display."""
        if not self.current_state:
            return Table()

        state = self.current_state

        # Create main table
        table = Table(title=f"Task Progress: {state.task_id}")

        # Add rows
        table.add_row("Task", state.task_description)
        table.add_row("Status", self._format_status(state.status))
        table.add_row(
            "Progress",
            f"{state.completed_subtasks}/{state.total_subtasks} subtasks ({state.progress_percentage:.1f}%)",
        )
        table.add_row(
            "Phases",
            f"{state.completed_phases}/{state.total_phases} ({state.phase_percentage:.1f}%)",
        )
        table.add_row("Current Phase", str(state.current_phase + 1))

        # Subtask breakdown
        table.add_row(
            "Subtasks",
            f"✓ {state.completed_subtasks} | ✗ {state.failed_subtasks} | ⊗ {state.blocked_subtasks}",
        )

        # Timing
        elapsed = state.elapsed_time
        table.add_row("Elapsed Time", str(elapsed).split(".")[0])

        if state.estimated_completion:
            remaining = state.estimated_completion - datetime.now()
            if remaining.total_seconds() > 0:
                table.add_row("Estimated Remaining", str(remaining).split(".")[0])
                table.add_row(
                    "Estimated Completion",
                    state.estimated_completion.strftime("%H:%M:%S"),
                )

        # Retries
        if state.retry_attempts:
            total_retries = sum(state.retry_attempts.values())
            table.add_row("Total Retries", str(total_retries))

        # Checkpoint
        if state.checkpoint_path:
            table.add_row("Latest Checkpoint", state.checkpoint_path)

        return table

    def _format_status(self, status: TaskStatus) -> str:
        """Format status with color."""
        if status == TaskStatus.COMPLETED:
            return "[green]✓ COMPLETED[/green]"
        elif status == TaskStatus.IN_PROGRESS:
            return "[cyan]⟳ IN PROGRESS[/cyan]"
        elif status == TaskStatus.FAILED:
            return "[red]✗ FAILED[/red]"
        elif status == TaskStatus.BLOCKED:
            return "[yellow]⊗ BLOCKED[/yellow]"
        else:
            return "[dim]PENDING[/dim]"

    def list_all_progress(self) -> list[ProgressState]:
        """
        List all progress files.

        Returns:
            List of progress states
        """
        states = []
        for progress_file in self.progress_dir.glob("*.json"):
            try:
                with open(progress_file) as f:
                    data = json.load(f)
                states.append(ProgressState.from_dict(data))
            except Exception:
                continue

        return sorted(states, key=lambda s: s.start_time, reverse=True)

    def cleanup_completed(self, days: int = 7) -> int:
        """
        Remove progress files for completed tasks older than N days.

        Args:
            days: Age threshold in days

        Returns:
            Number of files removed
        """
        cutoff = datetime.now() - timedelta(days=days)
        removed = 0

        for progress_file in self.progress_dir.glob("*.json"):
            try:
                with open(progress_file) as f:
                    data = json.load(f)

                state = ProgressState.from_dict(data)

                # Remove if completed and old
                if state.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.BLOCKED):
                    if state.last_update < cutoff:
                        progress_file.unlink()
                        removed += 1
            except Exception:
                continue

        return removed
