"""File writing utilities for applying agent changes."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

if TYPE_CHECKING:
    from modular_agents.core.models import FileChange, SubTaskResult

console = Console()


class FileWriter:
    """Handles writing file changes to disk with validation and user approval."""

    @staticmethod
    def validate_changes(
        changes: list[FileChange],
        module_name: str,
        expected_language: str,
        repo_root: Path,
    ) -> tuple[bool, list[str]]:
        """Validate file changes before writing.

        Returns:
            (is_valid, errors) tuple
        """
        errors = []

        # Language validation
        language_extensions = {
            "scala": [".scala"],
            "python": [".py"],
            "java": [".java"],
            "javascript": [".js", ".ts", ".jsx", ".tsx"],
            "typescript": [".ts", ".tsx"],
            "go": [".go"],
            "rust": [".rs"],
        }

        expected_exts = language_extensions.get(expected_language.lower(), [])

        if expected_exts:
            for change in changes:
                # Skip non-code files
                if any(change.path.endswith(ext) for ext in [".json", ".yaml", ".yml", ".md", ".txt", ".xml", ".conf", ".properties"]):
                    continue

                # Check language
                if not any(change.path.endswith(ext) for ext in expected_exts):
                    errors.append(
                        f"Language violation: {change.path} - "
                        f"Expected {expected_language} files {expected_exts}"
                    )

        # Path validation
        for change in changes:
            path = Path(change.path)

            # Ensure absolute path
            if not path.is_absolute():
                path = repo_root / change.path

            # Check path traversal
            try:
                path.resolve().relative_to(repo_root.resolve())
            except ValueError:
                errors.append(f"Security: {change.path} is outside repository")

        return (len(errors) == 0, errors)

    @staticmethod
    def preview_changes(changes: list[FileChange], module_name: str) -> None:
        """Display a preview of changes to be made."""
        console.print(f"\n[bold cyan]Preview Changes for Module: {module_name}[/bold cyan]\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Action", style="cyan", width=10)
        table.add_column("File", width=60)
        table.add_column("Size", justify="right", width=10)

        for change in changes:
            action_color = {
                "create": "green",
                "modify": "yellow",
                "delete": "red",
            }.get(change.action, "white")

            size = f"{len(change.content or '')} bytes" if change.content else "-"

            table.add_row(
                f"[{action_color}]{change.action.upper()}[/{action_color}]",
                change.path,
                size
            )

        console.print(table)

    @staticmethod
    def apply_changes(
        changes: list[FileChange],
        repo_root: Path,
        dry_run: bool = False,
        interactive: bool = True,
    ) -> tuple[int, int, list[str]]:
        """Apply file changes to disk.

        Args:
            changes: List of file changes to apply
            repo_root: Repository root path
            dry_run: If True, don't actually write files
            interactive: If True, ask for confirmation

        Returns:
            (success_count, failure_count, errors) tuple
        """
        if not changes:
            return (0, 0, [])

        # Preview changes
        FileWriter.preview_changes(changes, "module")

        if interactive and not dry_run:
            if not Confirm.ask("\nApply these changes?", default=False):
                console.print("[yellow]Changes cancelled by user[/yellow]")
                return (0, 0, ["Cancelled by user"])

        if dry_run:
            console.print("\n[yellow]DRY RUN - No files will be written[/yellow]")
            return (len(changes), 0, [])

        success_count = 0
        failure_count = 0
        errors = []

        for change in changes:
            try:
                path = Path(change.path)
                if not path.is_absolute():
                    path = repo_root / change.path

                if change.action == "create" or change.action == "modify":
                    # Create parent directories
                    path.parent.mkdir(parents=True, exist_ok=True)

                    # Write content
                    if change.content:
                        path.write_text(change.content)
                        console.print(f"[green]✓[/green] {change.action}: {path.relative_to(repo_root)}")
                        success_count += 1
                    else:
                        errors.append(f"No content for {path}")
                        failure_count += 1

                elif change.action == "delete":
                    if path.exists():
                        path.unlink()
                        console.print(f"[red]✓[/red] deleted: {path.relative_to(repo_root)}")
                        success_count += 1
                    else:
                        console.print(f"[yellow]⚠[/yellow] {path} doesn't exist (skip)")

            except Exception as e:
                error_msg = f"Failed to {change.action} {change.path}: {e}"
                errors.append(error_msg)
                console.print(f"[red]✗[/red] {error_msg}")
                failure_count += 1

        return (success_count, failure_count, errors)

    @staticmethod
    def apply_subtask_results(
        results: list[SubTaskResult],
        repo_root: Path,
        module_profiles: dict,
        dry_run: bool = False,
        interactive: bool = True,
    ) -> dict:
        """Apply all subtask results to disk.

        Returns:
            Summary dict with counts and errors
        """
        summary = {
            "total_files": 0,
            "created": 0,
            "modified": 0,
            "deleted": 0,
            "failed": 0,
            "errors": [],
            "validation_failures": [],
        }

        for result in results:
            if not result.changes:
                continue

            # Get module info for validation
            module_name = result.subtask_id.split("_")[0] if "_" in result.subtask_id else "unknown"
            module_profile = module_profiles.get(module_name)

            if module_profile:
                # Validate changes
                is_valid, errors = FileWriter.validate_changes(
                    result.changes,
                    module_name,
                    module_profile.language,
                    repo_root,
                )

                if not is_valid:
                    console.print(f"\n[red]Validation failed for {module_name}:[/red]")
                    for error in errors:
                        console.print(f"  [red]✗[/red] {error}")
                    summary["validation_failures"].extend(errors)
                    continue

            # Apply changes
            success, failed, file_errors = FileWriter.apply_changes(
                result.changes,
                repo_root,
                dry_run=dry_run,
                interactive=interactive,
            )

            summary["total_files"] += len(result.changes)
            summary["failed"] += failed
            summary["errors"].extend(file_errors)

            # Count by action
            for change in result.changes[:success]:
                if change.action == "create":
                    summary["created"] += 1
                elif change.action == "modify":
                    summary["modified"] += 1
                elif change.action == "delete":
                    summary["deleted"] += 1

        return summary
