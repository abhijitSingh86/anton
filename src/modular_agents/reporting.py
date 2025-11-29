"""Reporting and summary generation for agent execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

if TYPE_CHECKING:
    from modular_agents.core.models import TaskResult, SubTaskResult

console = Console()


class AgentSummaryReporter:
    """Generate comprehensive summaries of agent activities."""

    @staticmethod
    def generate_task_summary(result: TaskResult) -> str:
        """Generate a comprehensive task summary."""
        lines = []

        lines.append(f"# Task Execution Summary")
        lines.append(f"\nTask ID: {result.task_id}")
        lines.append(f"Status: {result.status.value}")

        if result.error:
            lines.append(f"\nError: {result.error}")

        # Group results by module
        module_results: dict[str, list[SubTaskResult]] = {}
        for sr in result.subtask_results:
            # Extract module name from subtask_id (format: taskid_st_N)
            # We need to look up which module this belongs to
            module = sr.subtask_id.split('_')[0] if '_' in sr.subtask_id else "unknown"
            if module not in module_results:
                module_results[module] = []
            module_results[module].append(sr)

        lines.append(f"\n## Agent Activities ({len(module_results)} modules involved)")

        for module, results in module_results.items():
            lines.append(f"\n### Module: {module}")
            for sr in results:
                lines.append(f"  - Subtask {sr.subtask_id}: {sr.status.value}")
                if sr.changes:
                    lines.append(f"    Files changed: {len(sr.changes)}")
                if sr.tests_added:
                    lines.append(f"    Tests added: {len(sr.tests_added)}")
                if sr.error:
                    lines.append(f"    Error: {sr.error}")

        return "\n".join(lines)

    @staticmethod
    def display_agent_summary(result: TaskResult, module_map: dict[str, str] | None = None) -> None:
        """Display a rich formatted summary of what each agent learned/did.

        Args:
            result: Task result with subtask results
            module_map: Optional mapping of subtask_id to module name
        """
        console.print("\n")
        console.print(Panel.fit(
            "[bold cyan]Agent Learning Summary[/bold cyan]",
            border_style="cyan"
        ))

        # Group by module
        from collections import defaultdict
        module_results: dict[str, list[SubTaskResult]] = defaultdict(list)

        for sr in result.subtask_results:
            # Try to extract module from subtask if module_map provided
            if module_map and sr.subtask_id in module_map:
                module = module_map[sr.subtask_id]
            else:
                # Parse from notes or use unknown
                module = "unknown"

            module_results[module].append(sr)

        # Create summary table
        table = Table(title="Module Agent Activities", show_header=True, header_style="bold magenta")
        table.add_column("Agent", style="cyan", width=20)
        table.add_column("Status", width=12)
        table.add_column("Files", justify="right", width=8)
        table.add_column("Tests", justify="right", width=8)
        table.add_column("Summary", width=50)

        total_files = 0
        total_tests = 0

        for module, results in sorted(module_results.items()):
            for sr in results:
                # Status with emoji
                status_display = {
                    "completed": "[green]✓ Completed[/green]",
                    "failed": "[red]✗ Failed[/red]",
                    "blocked": "[yellow]⊗ Blocked[/yellow]",
                    "pending": "[dim]○ Pending[/dim]",
                }.get(sr.status.value, sr.status.value)

                files_changed = len(sr.changes)
                tests_added = len(sr.tests_added)
                total_files += files_changed
                total_tests += tests_added

                # Create summary from notes or changes
                summary = sr.notes[:50] if sr.notes else ""
                if not summary and sr.changes:
                    actions = [c.action for c in sr.changes]
                    summary = f"{len(actions)} changes: {', '.join(set(actions))}"
                if sr.error:
                    summary = f"[red]{sr.error[:50]}[/red]"

                table.add_row(
                    f"[cyan]{module}[/cyan]",
                    status_display,
                    str(files_changed) if files_changed > 0 else "-",
                    str(tests_added) if tests_added > 0 else "-",
                    summary + "..." if len(sr.notes or "") > 50 else summary
                )

        console.print(table)

        # Summary statistics
        console.print(f"\n[bold]Overall Statistics:[/bold]")
        console.print(f"  Total files changed: {total_files}")
        console.print(f"  Total tests added: {total_tests}")
        console.print(f"  Modules involved: {len(module_results)}")

        completed = sum(1 for sr in result.subtask_results if sr.status.value == "completed")
        console.print(f"  Success rate: {completed}/{len(result.subtask_results)} subtasks")

    @staticmethod
    def display_detailed_agent_learning(result: TaskResult, module_map: dict[str, str] | None = None) -> None:
        """Display detailed tree view of what each agent learned."""
        tree = Tree("[bold cyan]Agent Learning & Knowledge Acquired[/bold cyan]")

        # Group by module
        from collections import defaultdict
        module_results: dict[str, list[SubTaskResult]] = defaultdict(list)

        for sr in result.subtask_results:
            module = module_map.get(sr.subtask_id, "unknown") if module_map else "unknown"
            module_results[module].append(sr)

        for module, results in sorted(module_results.items()):
            module_branch = tree.add(f"[bold cyan]Agent: {module}[/bold cyan]")

            for sr in results:
                status_icon = {
                    "completed": "✓",
                    "failed": "✗",
                    "blocked": "⊗",
                }.get(sr.status.value, "○")

                subtask_branch = module_branch.add(
                    f"{status_icon} Subtask: {sr.subtask_id} [{sr.status.value}]"
                )

                # Show what was learned/accomplished
                if sr.changes:
                    changes_branch = subtask_branch.add(f"[green]Files Modified ({len(sr.changes)})[/green]")
                    for change in sr.changes[:5]:  # Limit to first 5
                        action_color = {
                            "create": "green",
                            "modify": "yellow",
                            "delete": "red",
                        }.get(change.action, "white")
                        changes_branch.add(f"[{action_color}]{change.action}:[/{action_color}] {change.path}")
                    if len(sr.changes) > 5:
                        changes_branch.add(f"[dim]... and {len(sr.changes) - 5} more[/dim]")

                if sr.tests_added:
                    tests_branch = subtask_branch.add(f"[blue]Tests Added ({len(sr.tests_added)})[/blue]")
                    for test in sr.tests_added[:3]:
                        tests_branch.add(f"📝 {test}")
                    if len(sr.tests_added) > 3:
                        tests_branch.add(f"[dim]... and {len(sr.tests_added) - 3} more[/dim]")

                if sr.notes:
                    notes_branch = subtask_branch.add("[yellow]Implementation Notes[/yellow]")
                    # Split notes into lines and show first few
                    note_lines = sr.notes.split('\n')[:3]
                    for line in note_lines:
                        if line.strip():
                            notes_branch.add(f"💡 {line.strip()[:80]}")

                if sr.error:
                    error_branch = subtask_branch.add("[red]Errors Encountered[/red]")
                    error_branch.add(f"❌ {sr.error}")

                if sr.blockers:
                    blocker_branch = subtask_branch.add("[yellow]Blockers Reported[/yellow]")
                    for blocker in sr.blockers:
                        blocker_branch.add(f"🚧 {blocker}")

        console.print("\n")
        console.print(tree)

    @staticmethod
    def save_learning_summary(result: TaskResult, filepath: str, module_map: dict[str, str] | None = None) -> None:
        """Save agent learning summary to a markdown file."""
        from pathlib import Path
        from datetime import datetime

        lines = [
            "# Agent Learning Summary",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nTask ID: {result.task_id}",
            f"Overall Status: **{result.status.value}**",
            "",
        ]

        if result.summary:
            lines.extend([
                "## Task Summary",
                "",
                result.summary,
                "",
            ])

        # Group by module
        from collections import defaultdict
        module_results: dict[str, list[SubTaskResult]] = defaultdict(list)

        for sr in result.subtask_results:
            module = module_map.get(sr.subtask_id, "unknown") if module_map else "unknown"
            module_results[module].append(sr)

        lines.append("## Agent Activities\n")

        for module, results in sorted(module_results.items()):
            lines.append(f"### Agent: `{module}`\n")

            for sr in results:
                status_emoji = {
                    "completed": "✅",
                    "failed": "❌",
                    "blocked": "🚧",
                }.get(sr.status.value, "⚪")

                lines.append(f"#### {status_emoji} Subtask: `{sr.subtask_id}`\n")
                lines.append(f"**Status**: {sr.status.value}\n")

                if sr.changes:
                    lines.append(f"**Files Changed**: {len(sr.changes)}\n")
                    for change in sr.changes:
                        lines.append(f"- `{change.action}`: {change.path}")
                    lines.append("")

                if sr.tests_added:
                    lines.append(f"**Tests Added**: {len(sr.tests_added)}\n")
                    for test in sr.tests_added:
                        lines.append(f"- {test}")
                    lines.append("")

                if sr.notes:
                    lines.append("**Notes**:\n")
                    lines.append(f"```\n{sr.notes}\n```\n")

                if sr.error:
                    lines.append(f"**Error**: {sr.error}\n")

                if sr.blockers:
                    lines.append("**Blockers**:\n")
                    for blocker in sr.blockers:
                        lines.append(f"- {blocker}")
                    lines.append("")

                lines.append("---\n")

        # Statistics
        total_files = sum(len(sr.changes) for sr in result.subtask_results)
        total_tests = sum(len(sr.tests_added) for sr in result.subtask_results)
        completed = sum(1 for sr in result.subtask_results if sr.status.value == "completed")

        lines.extend([
            "## Statistics\n",
            f"- **Total files changed**: {total_files}",
            f"- **Total tests added**: {total_tests}",
            f"- **Modules involved**: {len(module_results)}",
            f"- **Success rate**: {completed}/{len(result.subtask_results)} subtasks completed",
            "",
        ])

        Path(filepath).write_text("\n".join(lines))
        console.print(f"\n[green]Learning summary saved to {filepath}[/green]")
