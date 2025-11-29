"""Autonomy management and safety checks for agent execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from modular_agents.core.models import FileChange, SubTaskResult


# =============================================================================
# Autonomy Configuration
# =============================================================================


class AutonomyLevel(str, Enum):
    """Agent autonomy levels."""

    INTERACTIVE = "interactive"  # Ask before every action (default)
    SUPERVISED = "supervised"  # Auto-approve safe actions, ask for risky
    AUTONOMOUS = "autonomous"  # Auto-approve all with safety checks
    FULL = "full"  # No approval, full autonomy (with safety checks)


@dataclass
class SafetyChecks:
    """Safety check configuration."""

    max_files_per_subtask: int = 10
    max_lines_per_file: int = 1000
    max_total_lines: int = 5000
    forbidden_paths: list[str] = field(
        default_factory=lambda: [
            ".git/",
            ".env",
            ".env.local",
            ".env.production",
            "secrets/",
            "credentials.json",
            "config/secrets",
        ]
    )
    forbidden_actions: list[str] = field(default_factory=lambda: ["delete_all"])
    warn_on_patterns: list[str] = field(
        default_factory=lambda: [
            "API_KEY",
            "SECRET",
            "PASSWORD",
            "TOKEN",
            "PRIVATE_KEY",
        ]
    )


@dataclass
class AutonomyConfig:
    """Configuration for agent autonomy."""

    level: AutonomyLevel = AutonomyLevel.INTERACTIVE
    auto_retry: bool = False
    max_autonomous_retries: int = 3
    auto_checkpoint: bool = True
    auto_resume: bool = False
    require_approval_for: list[str] = field(
        default_factory=lambda: [
            "file_deletion",
            "large_changes",  # > 500 lines
            "config_changes",
            "dependency_changes",
            "multiple_modules",  # Changes across >3 modules
        ]
    )
    safety_checks: SafetyChecks = field(default_factory=SafetyChecks)
    timeout_seconds: int = 600  # 10 minutes per subtask


# =============================================================================
# Safety Violation
# =============================================================================


@dataclass
class SafetyViolation:
    """Represents a safety check violation."""

    check: str
    severity: str  # "warning", "error", "critical"
    message: str
    affected_files: list[str] = field(default_factory=list)
    recommendation: str = ""


# =============================================================================
# Autonomy Manager
# =============================================================================


class AutonomyManager:
    """Manages agent autonomy and approval workflow."""

    def __init__(self, config: AutonomyConfig | None = None):
        """
        Initialize autonomy manager.

        Args:
            config: Autonomy configuration
        """
        self.config = config or AutonomyConfig()

    def should_request_approval(
        self, action: str, context: dict[str, Any] | None = None
    ) -> bool:
        """
        Check if action requires user approval.

        Args:
            action: Action type (e.g., "file_deletion", "large_changes")
            context: Optional context about the action

        Returns:
            True if approval required
        """
        context = context or {}

        # FULL level never requires approval
        if self.config.level == AutonomyLevel.FULL:
            return False

        # INTERACTIVE always requires approval
        if self.config.level == AutonomyLevel.INTERACTIVE:
            return True

        # SUPERVISED and AUTONOMOUS check specific actions
        if action in self.config.require_approval_for:
            return True

        # SUPERVISED checks additional criteria
        if self.config.level == AutonomyLevel.SUPERVISED:
            # Check file count
            file_count = context.get("file_count", 0)
            if file_count > 5:
                return True

            # Check line count
            line_count = context.get("line_count", 0)
            if line_count > 500:
                return True

            # Check if modifying multiple modules
            module_count = context.get("module_count", 0)
            if module_count > 3:
                return True

        return False

    def validate_safety_checks(
        self, changes: list[FileChange]
    ) -> tuple[bool, list[SafetyViolation]]:
        """
        Validate changes against safety rules.

        Args:
            changes: List of file changes

        Returns:
            Tuple of (is_safe, violations)
        """
        violations: list[SafetyViolation] = []

        # Check file count
        if len(changes) > self.config.safety_checks.max_files_per_subtask:
            violations.append(
                SafetyViolation(
                    check="max_files",
                    severity="warning",
                    message=f"Too many files modified: {len(changes)} > {self.config.safety_checks.max_files_per_subtask}",
                    recommendation="Consider breaking into smaller subtasks",
                )
            )

        total_lines = 0
        for change in changes:
            # Check forbidden paths
            for forbidden in self.config.safety_checks.forbidden_paths:
                if forbidden in change.path:
                    violations.append(
                        SafetyViolation(
                            check="forbidden_path",
                            severity="critical",
                            message=f"Attempt to modify forbidden path: {change.path}",
                            affected_files=[change.path],
                            recommendation="This path is protected and should not be modified",
                        )
                    )

            # Check line count
            if change.content:
                line_count = len(change.content.splitlines())
                total_lines += line_count

                if line_count > self.config.safety_checks.max_lines_per_file:
                    violations.append(
                        SafetyViolation(
                            check="max_lines_per_file",
                            severity="warning",
                            message=f"Large file change: {change.path} ({line_count} lines)",
                            affected_files=[change.path],
                            recommendation="Review if this change is necessary",
                        )
                    )

                # Check for sensitive patterns
                for pattern in self.config.safety_checks.warn_on_patterns:
                    if pattern in change.content:
                        violations.append(
                            SafetyViolation(
                                check="sensitive_content",
                                severity="warning",
                                message=f"Sensitive pattern '{pattern}' found in {change.path}",
                                affected_files=[change.path],
                                recommendation="Ensure no secrets are being committed",
                            )
                        )

            # Check forbidden actions
            if change.action in self.config.safety_checks.forbidden_actions:
                violations.append(
                    SafetyViolation(
                        check="forbidden_action",
                        severity="critical",
                        message=f"Forbidden action: {change.action}",
                        affected_files=[change.path],
                        recommendation="This action is not allowed",
                    )
                )

        # Check total lines
        if total_lines > self.config.safety_checks.max_total_lines:
            violations.append(
                SafetyViolation(
                    check="max_total_lines",
                    severity="warning",
                    message=f"Total lines changed: {total_lines} > {self.config.safety_checks.max_total_lines}",
                    recommendation="Consider breaking into multiple subtasks",
                )
            )

        # Check if any critical violations
        critical = any(v.severity == "critical" for v in violations)

        return (not critical, violations)

    def auto_approve_changes(
        self, changes: list[FileChange], context: dict[str, Any] | None = None
    ) -> tuple[bool, str]:
        """
        Auto-approve changes if safe and within autonomy level.

        Args:
            changes: List of file changes
            context: Optional context

        Returns:
            Tuple of (approved, reason)
        """
        context = context or {}

        # Always run safety checks
        is_safe, violations = self.validate_safety_checks(changes)

        # Critical violations always block
        critical_violations = [v for v in violations if v.severity == "critical"]
        if critical_violations:
            reasons = [v.message for v in critical_violations]
            return False, "Critical safety violations: " + "; ".join(reasons)

        # Check autonomy level
        if self.config.level == AutonomyLevel.INTERACTIVE:
            return False, "Interactive mode requires approval for all changes"

        # Check if approval needed based on action type
        action_type = self._classify_changes(changes)
        if self.should_request_approval(action_type, context):
            return False, f"Approval required for: {action_type}"

        # Auto-approve
        return True, "Auto-approved within autonomy level and safety checks"

    def _classify_changes(self, changes: list[FileChange]) -> str:
        """
        Classify changes to determine action type.

        Args:
            changes: List of file changes

        Returns:
            Action type string
        """
        # Check for deletions
        if any(c.action == "delete" for c in changes):
            return "file_deletion"

        # Check for large changes
        total_lines = sum(
            len(c.content.splitlines()) if c.content else 0 for c in changes
        )
        if total_lines > 500:
            return "large_changes"

        # Check for config changes
        config_extensions = [".json", ".yaml", ".yml", ".toml", ".ini", ".conf"]
        if any(
            any(c.path.endswith(ext) for ext in config_extensions) for c in changes
        ):
            return "config_changes"

        # Check for dependency changes
        dependency_files = [
            "package.json",
            "requirements.txt",
            "Cargo.toml",
            "build.sbt",
            "pom.xml",
            "build.gradle",
        ]
        if any(any(df in c.path for df in dependency_files) for c in changes):
            return "dependency_changes"

        # Check for multiple modules
        modules = set()
        for change in changes:
            # Extract module from path (assuming path format like "module/...")
            parts = Path(change.path).parts
            if len(parts) > 0:
                modules.add(parts[0])

        if len(modules) > 3:
            return "multiple_modules"

        return "normal_changes"

    def get_approval_prompt(
        self, result: SubTaskResult, violations: list[SafetyViolation]
    ) -> str:
        """
        Generate approval prompt for user.

        Args:
            result: Subtask result
            violations: List of safety violations

        Returns:
            Formatted approval prompt
        """
        lines = []
        lines.append(f"\n{'=' * 60}")
        lines.append("APPROVAL REQUIRED")
        lines.append(f"{'=' * 60}")
        lines.append(f"\nSubtask: {result.subtask_id}")
        lines.append(f"Status: {result.status}")
        lines.append(f"Files to change: {len(result.changes)}")

        if violations:
            lines.append(f"\n⚠ Safety Warnings ({len(violations)}):")
            for v in violations:
                lines.append(f"  - [{v.severity.upper()}] {v.message}")
                if v.recommendation:
                    lines.append(f"    → {v.recommendation}")

        lines.append("\nChanges:")
        for change in result.changes[:10]:  # Limit to first 10
            lines.append(f"  - {change.action.upper()}: {change.path}")

        if len(result.changes) > 10:
            lines.append(f"  ... and {len(result.changes) - 10} more")

        lines.append(f"\n{'=' * 60}")
        lines.append("Approve these changes? (yes/no): ")

        return "\n".join(lines)

    def check_timeout(self, start_time: float, current_time: float) -> bool:
        """
        Check if execution has exceeded timeout.

        Args:
            start_time: Start time (timestamp)
            current_time: Current time (timestamp)

        Returns:
            True if timed out
        """
        elapsed = current_time - start_time
        return elapsed > self.config.timeout_seconds

    def can_auto_retry(self, error_type: str, attempt: int) -> bool:
        """
        Check if can automatically retry.

        Args:
            error_type: Type of error
            attempt: Current attempt number

        Returns:
            True if can retry
        """
        if not self.config.auto_retry:
            return False

        if attempt >= self.config.max_autonomous_retries:
            return False

        # Only auto-retry for certain error types
        retriable_errors = [
            "LLM_TIMEOUT",
            "LLM_RATE_LIMIT",
            "NETWORK_ERROR",
            "TRANSIENT_ERROR",
        ]

        return error_type in retriable_errors
