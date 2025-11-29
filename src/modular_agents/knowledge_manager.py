"""Knowledge persistence and enrichment for repository analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from modular_agents.core.models import ModuleProfile, ProjectType, RepoKnowledge

console = Console()


class KnowledgeManager:
    """Manage repository knowledge persistence and enrichment."""

    KNOWLEDGE_FILE = "knowledge.json"
    CUSTOM_FILE = "custom.json"
    AGENTS_DIR = ".modular-agents"

    @classmethod
    def get_knowledge_path(cls, repo_path: Path) -> Path:
        """Get the path to the knowledge file."""
        return repo_path / cls.AGENTS_DIR / cls.KNOWLEDGE_FILE

    @classmethod
    def get_custom_path(cls, repo_path: Path) -> Path:
        """Get the path to the custom metadata file."""
        return repo_path / cls.AGENTS_DIR / cls.CUSTOM_FILE

    @classmethod
    def save_knowledge(cls, knowledge: RepoKnowledge, repo_path: Path) -> None:
        """Save repository knowledge to disk."""
        knowledge_path = cls.get_knowledge_path(repo_path)
        knowledge_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict for JSON serialization
        data = knowledge.model_dump(mode='json')

        # Convert Path objects to strings
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(v) for v in obj]
            elif isinstance(obj, Path):
                return str(obj)
            return obj

        data = convert_paths(data)
        knowledge_path.write_text(json.dumps(data, indent=2, default=str))
        console.print(f"[green]✓[/green] Knowledge saved to {knowledge_path}")

    @classmethod
    def load_knowledge(cls, repo_path: Path) -> RepoKnowledge | None:
        """Load repository knowledge from disk."""
        knowledge_path = cls.get_knowledge_path(repo_path)

        if not knowledge_path.exists():
            return None

        data = json.loads(knowledge_path.read_text())

        # Convert path strings back to Path objects
        data['root_path'] = Path(data['root_path'])
        for m in data.get('modules', []):
            m['path'] = Path(m['path'])

        return RepoKnowledge(**data)

    @classmethod
    def load_from_hierarchy(cls, repo_path: Path) -> RepoKnowledge | None:
        """Load knowledge from current dir or parent directories.

        This allows submodules to inherit knowledge from parent repos.
        """
        current = repo_path.resolve()

        # Check current directory first
        knowledge = cls.load_knowledge(current)
        if knowledge:
            console.print(f"[dim]Loaded knowledge from {current}[/dim]")
            return knowledge

        # Check parent directories (up to 3 levels)
        for _ in range(3):
            parent = current.parent
            if parent == current:  # Reached root
                break

            knowledge = cls.load_knowledge(parent)
            if knowledge:
                console.print(f"[dim]Loaded knowledge from parent: {parent}[/dim]")
                # Adjust paths to be relative to original repo_path
                return knowledge

            current = parent

        return None

    @classmethod
    def save_custom_metadata(cls, repo_path: Path, custom_data: dict[str, Any]) -> None:
        """Save user-provided custom metadata."""
        custom_path = cls.get_custom_path(repo_path)
        custom_path.parent.mkdir(parents=True, exist_ok=True)
        custom_path.write_text(json.dumps(custom_data, indent=2))
        console.print(f"[green]✓[/green] Custom metadata saved to {custom_path}")

    @classmethod
    def load_custom_metadata(cls, repo_path: Path) -> dict[str, Any]:
        """Load user-provided custom metadata."""
        custom_path = cls.get_custom_path(repo_path)

        if not custom_path.exists():
            return {}

        return json.loads(custom_path.read_text())

    @classmethod
    def enrich_knowledge(cls, knowledge: RepoKnowledge, repo_path: Path) -> RepoKnowledge:
        """Enrich auto-analyzed knowledge with user-provided custom data."""
        custom_data = cls.load_custom_metadata(repo_path)

        if not custom_data:
            return knowledge

        # Apply custom metadata to modules
        module_overrides = custom_data.get("modules", {})
        for module in knowledge.modules:
            if module.name in module_overrides:
                overrides = module_overrides[module.name]

                # Apply overrides
                if "purpose" in overrides:
                    module.purpose = overrides["purpose"]
                if "packages" in overrides:
                    module.packages = overrides["packages"]
                if "tags" in overrides:
                    # Add custom tags field if not exists
                    if not hasattr(module, "tags"):
                        module.tags = []
                    module.tags = overrides["tags"]

        # Apply repository-level metadata
        if "description" in custom_data:
            # Add to knowledge (would need to extend model)
            pass

        console.print("[dim]Applied custom metadata enrichments[/dim]")
        return knowledge

    @classmethod
    def interactive_enrich(cls, knowledge: RepoKnowledge, repo_path: Path) -> None:
        """Interactively enrich repository knowledge with user input."""
        console.print("\n")
        console.print(Panel.fit(
            "[bold cyan]Knowledge Enrichment[/bold cyan]\n"
            "Add custom information to improve agent understanding",
            border_style="cyan"
        ))

        if not Confirm.ask("Do you want to enrich module information?", default=False):
            return

        custom_data: dict[str, Any] = {"modules": {}}

        # Display modules table
        table = Table(title="Discovered Modules", show_header=True)
        table.add_column("#", style="cyan", width=4)
        table.add_column("Module", style="bold")
        table.add_column("Path", style="dim")
        table.add_column("Purpose", style="yellow")

        for i, module in enumerate(knowledge.modules, 1):
            table.add_row(
                str(i),
                module.name,
                str(module.path.relative_to(repo_path)),
                module.purpose or "[dim]Not specified[/dim]"
            )

        console.print(table)

        # Ask about each module
        for module in knowledge.modules:
            console.print(f"\n[bold cyan]Module: {module.name}[/bold cyan]")

            if Confirm.ask(f"Customize {module.name}?", default=False):
                module_custom: dict[str, Any] = {}

                # Purpose
                current_purpose = module.purpose or ""
                console.print(f"Current purpose: [yellow]{current_purpose}[/yellow]")
                new_purpose = Prompt.ask(
                    "New purpose (or press Enter to keep current)",
                    default=""
                )
                if new_purpose:
                    module_custom["purpose"] = new_purpose

                # Tags
                console.print("Add tags (comma-separated, e.g., 'api,public,critical')")
                tags_input = Prompt.ask("Tags", default="")
                if tags_input:
                    module_custom["tags"] = [t.strip() for t in tags_input.split(",")]

                # Additional notes
                notes = Prompt.ask("Additional notes for agents", default="")
                if notes:
                    module_custom["notes"] = notes

                if module_custom:
                    custom_data["modules"][module.name] = module_custom

        # Repository-level metadata
        console.print("\n[bold cyan]Repository-Level Information[/bold cyan]")
        repo_description = Prompt.ask(
            "Repository description (helps agents understand overall purpose)",
            default=""
        )
        if repo_description:
            custom_data["description"] = repo_description

        # Architecture notes
        arch_notes = Prompt.ask(
            "Architecture notes (key patterns, conventions, constraints)",
            default=""
        )
        if arch_notes:
            custom_data["architecture_notes"] = arch_notes

        # Save custom metadata
        if custom_data.get("modules") or custom_data.get("description"):
            cls.save_custom_metadata(repo_path, custom_data)
            console.print("\n[green]✓[/green] Enrichment complete!")
        else:
            console.print("\n[dim]No customizations provided[/dim]")

    @classmethod
    def display_knowledge_summary(cls, knowledge: RepoKnowledge) -> None:
        """Display a summary of loaded knowledge."""
        console.print("\n")
        console.print(Panel.fit(
            f"[bold]Repository: {knowledge.root_path.name}[/bold]\n"
            f"Type: {knowledge.project_type.value}\n"
            f"Modules: {len(knowledge.modules)}\n"
            f"Analyzed: {knowledge.analyzed_at.strftime('%Y-%m-%d %H:%M')}",
            title="[cyan]Knowledge Summary[/cyan]",
            border_style="cyan"
        ))

        # Module table
        table = Table(title="Modules", show_header=True)
        table.add_column("Module", style="cyan", width=20)
        table.add_column("Purpose", width=40)
        table.add_column("Files", justify="right", width=8)
        table.add_column("LOC", justify="right", width=8)

        for module in knowledge.modules:
            table.add_row(
                module.name,
                module.purpose[:37] + "..." if len(module.purpose) > 40 else module.purpose,
                str(module.file_count),
                str(module.loc)
            )

        console.print(table)

    @classmethod
    def export_knowledge(cls, knowledge: RepoKnowledge, output_path: Path) -> None:
        """Export knowledge to a human-readable markdown file."""
        lines = [
            f"# Repository Knowledge: {knowledge.root_path.name}",
            "",
            f"**Type**: {knowledge.project_type.value}",
            f"**Analyzed**: {knowledge.analyzed_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Modules**: {len(knowledge.modules)}",
            "",
            "## Modules",
            "",
        ]

        for module in knowledge.modules:
            lines.extend([
                f"### {module.name}",
                "",
                f"**Path**: `{module.path}`",
                f"**Purpose**: {module.purpose}",
                f"**Files**: {module.file_count} files, ~{module.loc} lines of code",
                "",
            ])

            if module.packages:
                lines.append("**Packages**:")
                for pkg in module.packages[:10]:
                    lines.append(f"- {pkg}")
                lines.append("")

            if module.dependencies:
                lines.append(f"**Dependencies**: {', '.join(module.dependencies)}")
                lines.append("")

            if module.dependents:
                lines.append(f"**Dependents**: {', '.join(module.dependents)}")
                lines.append("")

            lines.append("---")
            lines.append("")

        # Dependency graph
        if knowledge.dependency_graph:
            lines.extend([
                "## Dependency Graph",
                "",
                "```json",
                json.dumps(knowledge.dependency_graph, indent=2),
                "```",
                "",
            ])

        output_path.write_text("\n".join(lines))
        console.print(f"[green]✓[/green] Knowledge exported to {output_path}")
