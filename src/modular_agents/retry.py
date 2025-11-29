"""Retry management with exponential backoff for task execution."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


# =============================================================================
# Retry Configuration
# =============================================================================


class RetryStrategy(str, Enum):
    """Retry strategies."""

    IMMEDIATE = "immediate"  # No delay between retries
    LINEAR = "linear"  # Linear backoff (delay * attempt)
    EXPONENTIAL = "exponential"  # Exponential backoff (delay * factor^attempt)
    FIBONACCI = "fibonacci"  # Fibonacci backoff


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # Maximum delay cap
    backoff_factor: float = 2.0  # For exponential backoff
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    retry_on_errors: list[str] = field(
        default_factory=lambda: [
            "JSON_PARSE_ERROR",
            "LLM_TIMEOUT",
            "LLM_RATE_LIMIT",
            "NETWORK_ERROR",
            "TRANSIENT_ERROR",
        ]
    )
    retry_on_status: list[str] = field(
        default_factory=lambda: ["FAILED"]  # Retry on FAILED status
    )
    jitter: bool = True  # Add random jitter to delays


# =============================================================================
# Retry Attempt Tracking
# =============================================================================


@dataclass
class RetryAttempt:
    """Record of a retry attempt."""

    attempt_number: int
    timestamp: datetime
    error: str
    delay_before_next: float
    success: bool = False


@dataclass
class RetryHistory:
    """History of retry attempts for a subtask."""

    subtask_id: str
    task_id: str
    attempts: list[RetryAttempt] = field(default_factory=list)
    total_attempts: int = 0
    total_delay: float = 0.0
    final_success: bool = False
    final_error: str | None = None


# =============================================================================
# Retry Manager
# =============================================================================


class RetryManager:
    """Handles automatic retry with backoff."""

    def __init__(self, config: RetryConfig | None = None, log_dir: Path | None = None):
        """
        Initialize retry manager.

        Args:
            config: Retry configuration
            log_dir: Directory for retry logs (defaults to .modular-agents/retries)
        """
        self.config = config or RetryConfig()
        self.log_dir = log_dir or Path.cwd() / ".modular-agents" / "retries"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.histories: dict[str, RetryHistory] = {}

    async def retry_with_backoff(
        self,
        func: Callable[..., T],
        config: RetryConfig | None = None,
        context: dict[str, Any] | None = None,
    ) -> T:
        """
        Execute function with retry logic.

        Args:
            func: Async function to execute
            config: Optional retry config override
            context: Optional context (task_id, subtask_id, etc.)

        Returns:
            Result of successful function execution

        Raises:
            Last exception if all retries exhausted
        """
        config = config or self.config
        context = context or {}

        task_id = context.get("task_id", "unknown")
        subtask_id = context.get("subtask_id", "unknown")

        # Initialize history
        history = RetryHistory(
            subtask_id=subtask_id,
            task_id=task_id,
        )

        last_exception = None

        for attempt in range(config.max_retries + 1):  # +1 for initial attempt
            try:
                result = await func()

                # Success!
                if attempt > 0:
                    # Record successful retry
                    retry_attempt = RetryAttempt(
                        attempt_number=attempt,
                        timestamp=datetime.now(),
                        error="",
                        delay_before_next=0.0,
                        success=True,
                    )
                    history.attempts.append(retry_attempt)
                    history.total_attempts = attempt + 1
                    history.final_success = True

                    # Log success
                    self._log_retry_success(task_id, subtask_id, attempt, history)

                return result

            except Exception as e:
                last_exception = e
                error_name = self._classify_error(e)

                # Check if we should retry this error
                if not self._should_retry(error_name, config):
                    raise

                # Check if we have retries left
                if attempt >= config.max_retries:
                    # Record final failure
                    history.final_success = False
                    history.final_error = str(e)
                    history.total_attempts = attempt + 1
                    self._log_retry_failure(task_id, subtask_id, history)
                    raise

                # Calculate delay
                delay = self._calculate_delay(attempt, config)

                # Record attempt
                retry_attempt = RetryAttempt(
                    attempt_number=attempt,
                    timestamp=datetime.now(),
                    error=str(e),
                    delay_before_next=delay,
                    success=False,
                )
                history.attempts.append(retry_attempt)
                history.total_delay += delay

                # Log retry attempt
                self._log_retry_attempt(
                    task_id, subtask_id, attempt + 1, config.max_retries, error_name, delay
                )

                # Wait before retrying
                await asyncio.sleep(delay)

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError("Retry logic error: no result and no exception")

    def _should_retry(self, error_name: str, config: RetryConfig) -> bool:
        """
        Determine if error is retryable.

        Args:
            error_name: Classified error name
            config: Retry configuration

        Returns:
            True if should retry
        """
        return error_name in config.retry_on_errors

    def _classify_error(self, error: Exception) -> str:
        """
        Classify error for retry decision.

        Args:
            error: Exception instance

        Returns:
            Error classification string
        """
        error_str = str(error).lower()
        error_type = type(error).__name__

        # Check for specific patterns
        if "json" in error_str or "JSONDecodeError" in error_type:
            return "JSON_PARSE_ERROR"
        elif "timeout" in error_str or "TimeoutError" in error_type:
            return "LLM_TIMEOUT"
        elif "rate limit" in error_str or "429" in error_str:
            return "LLM_RATE_LIMIT"
        elif any(net in error_str for net in ["connection", "network", "dns"]):
            return "NETWORK_ERROR"
        elif "validation" in error_str:
            return "VALIDATION_ERROR"
        elif "permission" in error_str or "403" in error_str:
            return "PERMISSION_ERROR"
        else:
            return "TRANSIENT_ERROR"

    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """
        Calculate delay before next retry.

        Args:
            attempt: Current attempt number (0-indexed)
            config: Retry configuration

        Returns:
            Delay in seconds
        """
        if config.strategy == RetryStrategy.IMMEDIATE:
            delay = 0.0

        elif config.strategy == RetryStrategy.LINEAR:
            delay = config.initial_delay * (attempt + 1)

        elif config.strategy == RetryStrategy.EXPONENTIAL:
            delay = config.initial_delay * (config.backoff_factor**attempt)

        elif config.strategy == RetryStrategy.FIBONACCI:
            delay = config.initial_delay * self._fibonacci(attempt + 1)

        else:
            delay = config.initial_delay

        # Cap at max delay
        delay = min(delay, config.max_delay)

        # Add jitter if enabled
        if config.jitter:
            import random

            # Add up to 20% random jitter
            jitter_amount = delay * 0.2 * random.random()
            delay += jitter_amount

        return delay

    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

    def _log_retry_attempt(
        self,
        task_id: str,
        subtask_id: str,
        attempt: int,
        max_retries: int,
        error: str,
        delay: float,
    ) -> None:
        """Log a retry attempt."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "retry_attempt",
            "task_id": task_id,
            "subtask_id": subtask_id,
            "attempt": attempt,
            "max_retries": max_retries,
            "error": error,
            "delay_seconds": delay,
        }

        log_file = self.log_dir / f"{task_id}_retries.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def _log_retry_success(
        self, task_id: str, subtask_id: str, attempts: int, history: RetryHistory
    ) -> None:
        """Log successful retry."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "retry_success",
            "task_id": task_id,
            "subtask_id": subtask_id,
            "total_attempts": attempts + 1,
            "total_delay_seconds": history.total_delay,
        }

        log_file = self.log_dir / f"{task_id}_retries.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Also save detailed history
        self._save_history(task_id, subtask_id, history)

    def _log_retry_failure(
        self, task_id: str, subtask_id: str, history: RetryHistory
    ) -> None:
        """Log final retry failure."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "retry_failed",
            "task_id": task_id,
            "subtask_id": subtask_id,
            "total_attempts": history.total_attempts,
            "total_delay_seconds": history.total_delay,
            "final_error": history.final_error,
        }

        log_file = self.log_dir / f"{task_id}_retries.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Save detailed history
        self._save_history(task_id, subtask_id, history)

    def _save_history(self, task_id: str, subtask_id: str, history: RetryHistory) -> None:
        """Save detailed retry history."""
        history_file = self.log_dir / f"{task_id}_{subtask_id}_history.json"

        history_data = {
            "task_id": history.task_id,
            "subtask_id": history.subtask_id,
            "total_attempts": history.total_attempts,
            "total_delay_seconds": history.total_delay,
            "final_success": history.final_success,
            "final_error": history.final_error,
            "attempts": [
                {
                    "attempt_number": attempt.attempt_number,
                    "timestamp": attempt.timestamp.isoformat(),
                    "error": attempt.error,
                    "delay_before_next": attempt.delay_before_next,
                    "success": attempt.success,
                }
                for attempt in history.attempts
            ],
        }

        with open(history_file, "w") as f:
            json.dump(history_data, f, indent=2)

    def get_retry_stats(self, task_id: str) -> dict[str, Any]:
        """
        Get retry statistics for a task.

        Args:
            task_id: Task ID

        Returns:
            Dictionary with retry statistics
        """
        log_file = self.log_dir / f"{task_id}_retries.jsonl"
        if not log_file.exists():
            return {
                "total_retries": 0,
                "successful_retries": 0,
                "failed_retries": 0,
                "total_delay": 0.0,
            }

        total_retries = 0
        successful = 0
        failed = 0
        total_delay = 0.0

        with open(log_file) as f:
            for line in f:
                entry = json.loads(line)
                event = entry.get("event")

                if event == "retry_attempt":
                    total_retries += 1
                    total_delay += entry.get("delay_seconds", 0.0)
                elif event == "retry_success":
                    successful += 1
                elif event == "retry_failed":
                    failed += 1

        return {
            "total_retries": total_retries,
            "successful_retries": successful,
            "failed_retries": failed,
            "total_delay_seconds": total_delay,
        }
