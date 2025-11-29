"""Agent loop and runtime management."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from modular_agents.agents import ModuleAgent, OrchestratorAgent
from modular_agents.analyzers import AnalyzerRegistry
from modular_agents.core import RepoKnowledge, TaskResult, TaskStatus
from modular_agents.llm import LLMConfig, LLMProvider, LLMProviderRegistry

console = Console()


class AgentRuntime:
    """Runtime for the multi-agent system.
    
    Manages:
    - Repository analysis
    - Agent lifecycle
    - Task execution
    - State persistence
    """
    
    def __init__(
        self,
        repo_path: Path,
        llm_provider: str = "claude",
        llm_config: LLMConfig | None = None,
        knowledge_store: "KnowledgeStore | None" = None,
        checkpoint_manager: "CheckpointManager | None" = None,
        retry_manager: "RetryManager | None" = None,
        progress_tracker: "ProgressTracker | None" = None,
        autonomy_config: "AutonomyConfig | None" = None,
        autonomy_manager: "AutonomyManager | None" = None,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.llm_provider_name = llm_provider
        self.llm_config = llm_config
        self.knowledge_store = knowledge_store
        self.checkpoint_manager = checkpoint_manager
        self.retry_manager = retry_manager
        self.progress_tracker = progress_tracker
        self.autonomy_config = autonomy_config
        self.autonomy_manager = autonomy_manager

        self.repo_knowledge: RepoKnowledge | None = None
        self.llm: LLMProvider | None = None
        self.orchestrator: OrchestratorAgent | None = None
        self.module_agents: dict[str, ModuleAgent] = {}

        self._initialized = False
    
    async def initialize(self, force_reanalyze: bool = False) -> None:
        """Initialize the runtime - analyze repo and create agents."""
        if self._initialized and not force_reanalyze:
            return

        from modular_agents.knowledge_manager import KnowledgeManager

        # Try to load from hierarchy (current dir or parents)
        if not force_reanalyze:
            console.print("[dim]Looking for cached knowledge...[/dim]")
            self.repo_knowledge = KnowledgeManager.load_from_hierarchy(self.repo_path)

            if self.repo_knowledge:
                # Enrich with custom metadata if available
                self.repo_knowledge = KnowledgeManager.enrich_knowledge(
                    self.repo_knowledge, self.repo_path
                )

        # Analyze if no cached knowledge or force re-analyze
        if not self.repo_knowledge or force_reanalyze:
            console.print("[bold blue]Analyzing repository...[/bold blue]")
            self.repo_knowledge = await self._analyze_repo()
            KnowledgeManager.save_knowledge(self.repo_knowledge, self.repo_path)

            # Apply custom metadata if it exists
            self.repo_knowledge = KnowledgeManager.enrich_knowledge(
                self.repo_knowledge, self.repo_path
            )

        # Initialize LLM provider
        if self.llm_config:
            self.llm = LLMProviderRegistry.create(
                self.llm_provider_name,
                self.llm_config
            )

        # Create agents
        if self.llm:
            await self._create_agents()

        self._initialized = True

        console.print(f"[green]✓[/green] Initialized with {len(self.repo_knowledge.modules)} modules")
    
    async def _analyze_repo(self) -> RepoKnowledge:
        """Analyze the repository."""
        analyzer = AnalyzerRegistry.get_analyzer(self.repo_path)
        
        if not analyzer:
            raise ValueError(
                f"No analyzer found for {self.repo_path}. "
                f"Available: {AnalyzerRegistry.available()}"
            )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing project structure...", total=None)
            knowledge = await analyzer.analyze(self.repo_path)
            progress.update(task, description="Analysis complete")
        
        return knowledge
    
    async def _create_agents(self) -> None:
        """Create all agents."""
        if not self.repo_knowledge or not self.llm:
            return
        
        # Create module agents
        for module in self.repo_knowledge.modules:
            agent = ModuleAgent(
                profile=module,
                llm=self.llm,
                repo_root=self.repo_path,
            )
            self.module_agents[module.name] = agent
        
        # Create orchestrator
        self.orchestrator = OrchestratorAgent(
            repo_knowledge=self.repo_knowledge,
            module_agents=self.module_agents,
            llm=self.llm,
            knowledge_store=self.knowledge_store,
            checkpoint_manager=self.checkpoint_manager,
            retry_manager=self.retry_manager,
            progress_tracker=self.progress_tracker,
            autonomy_config=self.autonomy_config,
            autonomy_manager=self.autonomy_manager,
        )
    
    async def execute_task(
        self,
        task_description: str,
        resume_from: str | None = None,
    ) -> TaskResult:
        """Execute a task through the agent system.

        Args:
            task_description: The task to execute
            resume_from: Optional task ID to resume from checkpoint

        Returns:
            TaskResult with execution status and results
        """
        if not self._initialized:
            await self.initialize()

        if not self.orchestrator:
            raise RuntimeError("Orchestrator not initialized. Set LLM config.")

        console.print(Panel(
            task_description,
            title="[bold]Task[/bold]" if not resume_from else "[bold]Resuming Task[/bold]",
            border_style="blue",
        ))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Resuming from checkpoint..." if resume_from else "Processing task...",
                total=None,
            )
            result = await self.orchestrator.process_task(
                task_description,
                resume_from=resume_from,
            )
            progress.update(task, description="Task complete")

        # Store module map for reporting
        self.last_module_map = getattr(self.orchestrator, 'module_map', {})

        # Apply file changes if task completed successfully
        if result.status.value == "completed" and result.subtask_results:
            from modular_agents.file_writer import FileWriter

            console.print("\n[bold]Applying Changes to Disk[/bold]")

            # Build module profiles map
            module_profiles = {m.name: m for m in self.repo_knowledge.modules}

            summary = FileWriter.apply_subtask_results(
                result.subtask_results,
                self.repo_path,
                module_profiles,
                dry_run=False,
                interactive=True,  # Ask for confirmation
            )

            # Show summary
            if summary["validation_failures"]:
                console.print(f"\n[red]⚠ {len(summary['validation_failures'])} validation failures[/red]")
                result.error = f"Validation failures: {', '.join(summary['validation_failures'][:3])}"
                result.status = TaskStatus.FAILED

            if summary["total_files"] > 0:
                console.print(f"\n[green]File Changes Applied:[/green]")
                console.print(f"  Created: {summary['created']}")
                console.print(f"  Modified: {summary['modified']}")
                console.print(f"  Deleted: {summary['deleted']}")
                if summary['failed'] > 0:
                    console.print(f"  [red]Failed: {summary['failed']}[/red]")

        return result

    def get_module_map(self) -> dict[str, str]:
        """Get the module mapping from last task execution."""
        return getattr(self, 'last_module_map', {})
    
    def get_module_info(self, module_name: str | None = None) -> str:
        """Get information about modules."""
        if not self.repo_knowledge:
            return "Repository not analyzed yet."
        
        if module_name:
            module = self.repo_knowledge.get_module(module_name)
            if not module:
                return f"Module '{module_name}' not found."
            
            return f"""Module: {module.name}
Path: {module.path}
Purpose: {module.purpose}
Files: {module.file_count}
LOC: {module.loc}
Dependencies: {', '.join(module.dependencies) or 'None'}
Dependents: {', '.join(module.dependents) or 'None'}
Packages: {', '.join(module.packages[:5]) or 'None'}
"""
        
        # List all modules
        lines = ["Modules:\n"]
        for m in self.repo_knowledge.modules:
            lines.append(f"  {m.name}: {m.purpose}")
        return "\n".join(lines)
    


class InteractiveLoop:
    """Interactive REPL for the agent system."""
    
    def __init__(self, runtime: AgentRuntime):
        self.runtime = runtime
        self.commands: dict[str, Callable] = {
            "help": self._cmd_help,
            "modules": self._cmd_modules,
            "module": self._cmd_module,
            "task": self._cmd_task,
            "analyze": self._cmd_analyze,
            "quit": self._cmd_quit,
            "exit": self._cmd_quit,
        }
        self._running = True
    
    async def run(self) -> None:
        """Run the interactive loop."""
        console.print(Panel(
            "[bold]Modular Agents[/bold]\n"
            "Type 'help' for commands, or enter a task directly.",
            border_style="green",
        ))
        
        await self.runtime.initialize()
        
        while self._running:
            try:
                user_input = console.input("\n[bold blue]>[/bold blue] ").strip()
                
                if not user_input:
                    continue
                
                # Check for commands
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if cmd in self.commands:
                    await self.commands[cmd](args)
                else:
                    # Treat as a task
                    await self._cmd_task(user_input)
                    
            except KeyboardInterrupt:
                console.print("\n[dim]Use 'quit' to exit[/dim]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    async def _cmd_help(self, args: str) -> None:
        """Show help."""
        console.print("""
[bold]Commands:[/bold]
  help              Show this help
  modules           List all modules
  module <name>     Show module details
  task <desc>       Execute a task
  analyze           Re-analyze the repository
  quit              Exit
  
Or just type a task description directly.
""")
    
    async def _cmd_modules(self, args: str) -> None:
        """List modules."""
        console.print(self.runtime.get_module_info())
    
    async def _cmd_module(self, args: str) -> None:
        """Show module details."""
        if not args:
            console.print("[yellow]Usage: module <name>[/yellow]")
            return
        console.print(self.runtime.get_module_info(args))
    
    async def _cmd_task(self, args: str) -> None:
        """Execute a task."""
        if not args:
            console.print("[yellow]Please provide a task description[/yellow]")
            return

        result = await self.runtime.execute_task(args)

        # Display result
        status_color = {
            "completed": "green",
            "failed": "red",
            "blocked": "yellow",
        }.get(result.status.value, "white")

        console.print(f"\n[{status_color}]Status: {result.status.value}[/{status_color}]")

        if result.summary:
            console.print(Panel(result.summary, title="Summary"))

        if result.error:
            console.print(f"[red]Error: {result.error}[/red]")

        # Display agent learning summary
        from modular_agents.reporting import AgentSummaryReporter

        module_map = self.runtime.get_module_map()
        AgentSummaryReporter.display_agent_summary(result, module_map)

        # Display detailed tree view
        AgentSummaryReporter.display_detailed_agent_learning(result, module_map)

        # Save learning summary
        if result.subtask_results:
            summary_path = self.runtime.repo_path / ".modular-agents" / f"summary_{result.task_id}.md"
            AgentSummaryReporter.save_learning_summary(result, str(summary_path), module_map)
    
    async def _cmd_analyze(self, args: str) -> None:
        """Re-analyze repository."""
        await self.runtime.initialize(force_reanalyze=True)
    
    async def _cmd_quit(self, args: str) -> None:
        """Exit the loop."""
        self._running = False
        console.print("[dim]Goodbye![/dim]")
