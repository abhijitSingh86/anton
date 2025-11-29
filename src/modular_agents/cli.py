"""Command-line interface for modular-agents."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import click
from rich.console import Console
from rich.prompt import Confirm

from modular_agents.llm import LLMConfig, LLMProviderRegistry
from modular_agents.runtime import AgentRuntime, InteractiveLoop

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Modular Agents - Multi-agent framework for modular codebases."""
    pass


@main.command()
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option(
    "--provider", "-p",
    type=click.Choice(["claude", "openai", "ollama"]),
    default="claude",
    help="LLM provider to use",
)
@click.option(
    "--model", "-m",
    default=None,
    help="Model name (default depends on provider)",
)
@click.option(
    "--api-key", "-k",
    envvar="LLM_API_KEY",
    default=None,
    help="API key (or set LLM_API_KEY env var)",
)
@click.option(
    "--base-url", "-u",
    default=None,
    help="Base URL for API (for local servers)",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show LLM interactions in real-time",
)
@click.option(
    "--debug", "-d",
    is_flag=True,
    help="Show full debug output including prompts",
)
@click.option(
    "--use-knowledge", "-kb",
    is_flag=True,
    help="Enable knowledge base for code context (requires indexed repository)",
)
@click.option(
    "--kb-db",
    type=click.Path(),
    help="Knowledge base database path (default: ~/.modular-agents/knowledge.db)",
)
@click.option(
    "--embedding-model",
    default="sentence-transformers/all-MiniLM-L6-v2",
    help="Embedding model for knowledge base",
)
def run(
    path: str,
    provider: str,
    model: str,
    api_key: str,
    base_url: str,
    verbose: bool,
    debug: bool,
    use_knowledge: bool,
    kb_db: str,
    embedding_model: str,
):
    """Run the interactive agent loop.

    PATH is the repository to analyze (default: current directory).
    """
    from modular_agents.trace import init_trace_logger

    # Initialize trace logger
    repo_path = Path(path).resolve()
    trace_dir = repo_path / ".modular-agents" / "traces" if (verbose or debug) else None
    init_trace_logger(trace_dir=trace_dir, verbose=verbose, debug=debug)

    # Determine default model based on provider
    default_models = {
        "claude": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
        "ollama": "llama3.1",
    }

    model = model or default_models.get(provider, "gpt-4o")

    # Get API key from environment if not provided
    if not api_key and provider in ("claude", "openai"):
        env_vars = {
            "claude": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        api_key = os.environ.get(env_vars.get(provider, ""))

        if not api_key:
            console.print(
                f"[yellow]Warning: No API key provided. "
                f"Set {env_vars.get(provider)} or use --api-key[/yellow]"
            )

    # Create config
    config = LLMConfig(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )

    # Set up knowledge store if requested
    knowledge_store = None
    if use_knowledge:
        from modular_agents.knowledge.embeddings import GemmaEmbeddingProvider
        from modular_agents.knowledge.store import KnowledgeStore

        # Determine database path
        if kb_db:
            db_path = Path(kb_db)
        else:
            db_path = Path.home() / ".modular-agents" / "knowledge.db"

        if not db_path.exists():
            console.print(f"[yellow]Warning: Knowledge base not found at {db_path}[/yellow]")
            console.print("Run 'anton index .' to create the knowledge base.")
            console.print("Continuing without knowledge base...")
        else:
            try:
                embedding_provider = GemmaEmbeddingProvider(
                    model_name=embedding_model,
                    device="cpu",
                )
                knowledge_store = KnowledgeStore(db_path, embedding_provider)
                knowledge_store.connect()
                console.print(f"[green]✓[/green] Knowledge base enabled: {db_path}")
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to load knowledge base: {e}[/yellow]")
                console.print("Continuing without knowledge base...")

    # Create runtime
    runtime = AgentRuntime(
        repo_path=repo_path,
        llm_provider=provider,
        llm_config=config,
        knowledge_store=knowledge_store,
    )

    # Run interactive loop
    loop = InteractiveLoop(runtime)
    asyncio.run(loop.run())


@main.command()
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--force", "-f", is_flag=True, help="Force re-analysis even if knowledge exists")
@click.option("--enrich", "-e", is_flag=True, help="Interactively enrich with custom information")
@click.option("--export", "-x", type=click.Path(), help="Export knowledge to markdown file")
def init(path: str, force: bool, enrich: bool, export: str):
    """Initialize repository analysis and save knowledge.

    PATH is the repository to analyze (default: current directory).

    This command analyzes your repository structure, creates module profiles,
    and saves the knowledge for future use. You can optionally enrich the
    auto-generated knowledge with custom information.
    """
    import asyncio
    from modular_agents.analyzers import AnalyzerRegistry
    from modular_agents.knowledge_manager import KnowledgeManager

    repo_path = Path(path).resolve()

    # Check if knowledge already exists
    existing = KnowledgeManager.load_knowledge(repo_path)
    if existing and not force:
        console.print("[yellow]Knowledge already exists for this repository[/yellow]")
        KnowledgeManager.display_knowledge_summary(existing)

        if not Confirm.ask("Re-analyze anyway?", default=False):
            if enrich:
                KnowledgeManager.interactive_enrich(existing, repo_path)
            return

    # Analyze repository
    console.print(f"[bold blue]Analyzing repository:[/bold blue] {repo_path}")

    analyzer = AnalyzerRegistry.get_analyzer(repo_path)
    if not analyzer:
        console.print(f"[red]No analyzer found for {repo_path}[/red]")
        console.print(f"Available analyzers: {AnalyzerRegistry.available()}")
        return

    async def do_analyze():
        return await analyzer.analyze(repo_path)

    knowledge = asyncio.run(do_analyze())

    # Save knowledge
    KnowledgeManager.save_knowledge(knowledge, repo_path)

    # Display summary
    KnowledgeManager.display_knowledge_summary(knowledge)

    # Optional enrichment
    if enrich:
        KnowledgeManager.interactive_enrich(knowledge, repo_path)
        # Reload with enrichments
        knowledge = KnowledgeManager.enrich_knowledge(knowledge, repo_path)

    # Optional export
    if export:
        export_path = Path(export)
        KnowledgeManager.export_knowledge(knowledge, export_path)

    console.print(f"\n[green]✓[/green] Initialization complete!")
    console.print(f"[dim]Run 'anton run {path}' to start the agent loop[/dim]")


@main.command()
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def analyze(path: str, as_json: bool):
    """Analyze a repository and show its structure.
    
    PATH is the repository to analyze (default: current directory).
    """
    import json
    
    from modular_agents.analyzers import AnalyzerRegistry
    
    repo_path = Path(path).resolve()
    analyzer = AnalyzerRegistry.get_analyzer(repo_path)
    
    if not analyzer:
        console.print(f"[red]No analyzer found for {repo_path}[/red]")
        console.print(f"Available analyzers: {AnalyzerRegistry.available()}")
        return
    
    async def do_analyze():
        return await analyzer.analyze(repo_path)
    
    knowledge = asyncio.run(do_analyze())
    
    if as_json:
        data = knowledge.model_dump(mode='json')
        # Convert paths to strings
        data['root_path'] = str(data['root_path'])
        for m in data.get('modules', []):
            m['path'] = str(m['path'])
        console.print_json(json.dumps(data, indent=2, default=str))
    else:
        console.print(f"\n[bold]Repository:[/bold] {knowledge.root_path}")
        console.print(f"[bold]Type:[/bold] {knowledge.project_type.value}")
        console.print(f"[bold]Modules:[/bold] {len(knowledge.modules)}\n")
        
        for module in knowledge.modules:
            console.print(f"  [bold blue]{module.name}[/bold blue]")
            console.print(f"    Path: {module.path}")
            console.print(f"    Purpose: {module.purpose}")
            console.print(f"    Files: {module.file_count}, LOC: {module.loc}")
            if module.dependencies:
                console.print(f"    Dependencies: {', '.join(module.dependencies)}")
            console.print()


@main.command()
@click.argument("path", type=click.Path(exists=True), default=".")
@click.argument("task", required=True)
@click.option(
    "--provider", "-p",
    type=click.Choice(["claude", "openai", "ollama"]),
    default="claude",
    help="LLM provider to use",
)
@click.option("--model", "-m", default=None, help="Model name")
@click.option("--api-key", "-k", envvar="LLM_API_KEY", default=None)
@click.option("--dry-run", is_flag=True, help="Show plan without executing")
@click.option("--verbose", "-v", is_flag=True, help="Show LLM interactions")
@click.option("--debug", "-d", is_flag=True, help="Show full debug output")
@click.option("--autonomous", is_flag=True, help="Enable full autonomous mode (no approval required)")
@click.option(
    "--autonomy-level",
    type=click.Choice(["interactive", "supervised", "autonomous", "full"]),
    default="supervised",
    help="Autonomy level: interactive (ask always), supervised (ask for risky), autonomous (auto-approve with checks), full (no approval)",
)
@click.option("--auto-retry", is_flag=True, help="Automatically retry failed subtasks")
@click.option("--max-retries", type=int, default=3, help="Maximum retry attempts")
@click.option("--resume", type=str, help="Resume from task ID (checkpoint)")
@click.option("--no-checkpoint", is_flag=True, help="Disable automatic checkpointing")
@click.option("--use-knowledge", "-kb", is_flag=True, help="Enable knowledge base (auto-enabled if repo is indexed)")
@click.option("--kb-db", type=click.Path(), help="Knowledge base database path")
@click.option("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model")
def task(
    path: str,
    task: str,
    provider: str,
    model: str,
    api_key: str,
    dry_run: bool,
    verbose: bool,
    debug: bool,
    autonomous: bool,
    autonomy_level: str,
    auto_retry: bool,
    max_retries: int,
    resume: str,
    no_checkpoint: bool,
    use_knowledge: bool,
    kb_db: str,
    embedding_model: str,
):
    """Execute a single task.

    PATH is the repository.
    TASK is the task description.
    """
    from modular_agents.trace import get_trace_logger, init_trace_logger

    # Initialize trace logger
    repo_path = Path(path).resolve()
    trace_dir = repo_path / ".modular-agents" / "traces" if (verbose or debug) else None
    init_trace_logger(trace_dir=trace_dir, verbose=verbose, debug=debug)

    default_models = {
        "claude": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
        "ollama": "llama3.1",
    }

    model = model or default_models.get(provider)

    if not api_key and provider in ("claude", "openai"):
        env_vars = {"claude": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
        api_key = os.environ.get(env_vars.get(provider, ""))

    config = LLMConfig(model=model, api_key=api_key)

    # Initialize continuation managers (ENABLED BY DEFAULT for best UX)
    checkpoint_manager = None
    retry_manager = None
    progress_tracker = None
    autonomy_config = None
    autonomy_manager = None
    knowledge_store = None

    if not dry_run:
        from modular_agents.autonomy import AutonomyConfig, AutonomyLevel, AutonomyManager
        from modular_agents.checkpoint import CheckpointManager
        from modular_agents.progress import ProgressTracker
        from modular_agents.retry import RetryManager

        # Determine autonomy level
        if autonomous:
            level = AutonomyLevel.FULL
        else:
            level = AutonomyLevel(autonomy_level)

        # Create managers (SMART DEFAULTS: enabled by default)
        if not no_checkpoint:
            checkpoint_manager = CheckpointManager()

        if auto_retry:
            retry_manager = RetryManager()

        # Always create progress tracker
        progress_tracker = ProgressTracker()

        autonomy_config = AutonomyConfig(
            level=level,
            auto_retry=auto_retry,
            max_autonomous_retries=max_retries,
            auto_checkpoint=not no_checkpoint,
        )
        autonomy_manager = AutonomyManager(autonomy_config)

        # Auto-detect and enable knowledge base if repository is indexed
        if use_knowledge or not use_knowledge:  # Check if repo has been indexed
            try:
                from modular_agents.knowledge.embeddings import GemmaEmbeddingProvider
                from modular_agents.knowledge.store import KnowledgeStore

                # Determine database path
                if kb_db:
                    db_path = Path(kb_db)
                else:
                    db_path = Path.home() / ".modular-agents" / "knowledge.db"

                # Check if database exists and if current repo is indexed
                if db_path.exists():
                    # Try to connect and check if repo is indexed
                    try:
                        embedding_provider = GemmaEmbeddingProvider(
                            model_name=embedding_model,
                            device="cpu",
                        )
                        knowledge_store = KnowledgeStore(db_path, embedding_provider)
                        knowledge_store.connect()

                        # Check if current repo is indexed
                        repos = knowledge_store.list_repos()
                        repo_path_str = str(repo_path.resolve())
                        is_indexed = any(r.repo_path == repo_path_str for r in repos)

                        if is_indexed:
                            console.print(f"[dim]✓ Knowledge base enabled (repository is indexed)[/dim]")
                        else:
                            knowledge_store.close()
                            knowledge_store = None

                    except Exception:
                        knowledge_store = None
            except ImportError:
                pass  # Knowledge base dependencies not installed

    runtime = AgentRuntime(
        repo_path=repo_path,
        llm_provider=provider,
        llm_config=config,
        checkpoint_manager=checkpoint_manager,
        retry_manager=retry_manager,
        progress_tracker=progress_tracker,
        autonomy_config=autonomy_config,
        autonomy_manager=autonomy_manager,
        knowledge_store=knowledge_store,
    )

    async def execute():
        await runtime.initialize()

        if dry_run:
            # Just show what would be done
            console.print("[bold]Dry run - showing execution plan[/bold]\n")
            subtasks = await runtime.orchestrator.decompose_task("dry", task)
            for st in subtasks:
                console.print(f"  [{st.module}] {st.description}")
            return

        result = await runtime.execute_task(task, resume_from=resume)

        status_color = {"completed": "green", "failed": "red"}.get(
            result.status.value, "yellow"
        )
        console.print(f"\n[{status_color}]Status: {result.status.value}[/{status_color}]")

        if result.summary:
            console.print(f"\n{result.summary}")

        if result.error:
            console.print(f"\n[red]Error: {result.error}[/red]")

        # Display agent learning summary
        from modular_agents.reporting import AgentSummaryReporter

        module_map = runtime.get_module_map()
        AgentSummaryReporter.display_agent_summary(result, module_map)
        AgentSummaryReporter.display_detailed_agent_learning(result, module_map)

        # Save learning summary
        if result.subtask_results:
            summary_path = repo_path / ".modular-agents" / f"summary_{result.task_id}.md"
            AgentSummaryReporter.save_learning_summary(result, str(summary_path), module_map)

        # Save trace summary if tracing was enabled
        trace_logger = get_trace_logger()
        if trace_logger and trace_logger.trace_dir:
            trace_logger.save_summary()

    asyncio.run(execute())


@main.command()
@click.argument("task_id", required=True)
@click.option("--path", type=click.Path(exists=True), default=".", help="Repository path")
@click.option(
    "--provider", "-p",
    type=click.Choice(["claude", "openai", "ollama"]),
    default="claude",
    help="LLM provider to use",
)
@click.option("--model", "-m", default=None, help="Model name")
@click.option("--api-key", "-k", envvar="LLM_API_KEY", default=None)
@click.option("--verbose", "-v", is_flag=True, help="Show LLM interactions")
@click.option("--debug", "-d", is_flag=True, help="Show full debug output")
def resume(
    task_id: str,
    path: str,
    provider: str,
    model: str,
    api_key: str,
    verbose: bool,
    debug: bool,
):
    """Resume a task from checkpoint.

    TASK_ID is the ID of the task to resume.
    """
    from modular_agents.autonomy import AutonomyConfig, AutonomyLevel, AutonomyManager
    from modular_agents.checkpoint import CheckpointManager
    from modular_agents.progress import ProgressTracker
    from modular_agents.retry import RetryManager
    from modular_agents.trace import init_trace_logger

    repo_path = Path(path).resolve()

    # Initialize trace logger
    trace_dir = repo_path / ".modular-agents" / "traces" if (verbose or debug) else None
    init_trace_logger(trace_dir=trace_dir, verbose=verbose, debug=debug)

    # Check if checkpoint exists
    checkpoint_manager = CheckpointManager()
    checkpoints = checkpoint_manager.list_checkpoints(task_id)

    if not checkpoints:
        console.print(f"[red]No checkpoints found for task: {task_id}[/red]")
        console.print(f"Available checkpoints are stored in: {repo_path / '.modular-agents' / 'checkpoints'}")
        return

    # Load latest checkpoint
    console.print(f"[bold]Resuming task:[/bold] {task_id}")
    console.print(f"[dim]Found {len(checkpoints)} checkpoint(s)[/dim]\n")

    checkpoint = checkpoint_manager.load_checkpoint(task_id)
    if not checkpoint:
        console.print(f"[red]Failed to load checkpoint for task: {task_id}[/red]")
        return

    console.print(f"[green]✓[/green] Loaded checkpoint from phase {checkpoint.phase_index + 1}/{checkpoint.total_phases}")
    console.print(f"[dim]Completed subtasks: {len(checkpoint.completed_subtasks)}[/dim]")
    console.print(f"[dim]Pending subtasks: {len(checkpoint.pending_subtasks)}[/dim]\n")

    # Set up LLM
    default_models = {
        "claude": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
        "ollama": "llama3.1",
    }
    model = model or default_models.get(provider)

    if not api_key and provider in ("claude", "openai"):
        env_vars = {"claude": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
        api_key = os.environ.get(env_vars.get(provider, ""))

    config = LLMConfig(model=model, api_key=api_key)

    # Initialize managers
    retry_manager = RetryManager()
    progress_tracker = ProgressTracker()
    autonomy_config = AutonomyConfig(
        level=AutonomyLevel.SUPERVISED,
        auto_retry=True,
        max_autonomous_retries=3,
        auto_checkpoint=True,
    )
    autonomy_manager = AutonomyManager(autonomy_config)

    # Create runtime
    runtime = AgentRuntime(
        repo_path=repo_path,
        llm_provider=provider,
        llm_config=config,
        checkpoint_manager=checkpoint_manager,
        retry_manager=retry_manager,
        progress_tracker=progress_tracker,
        autonomy_config=autonomy_config,
        autonomy_manager=autonomy_manager,
    )

    async def do_resume():
        await runtime.initialize()

        # Resume execution
        result = await runtime.execute_task(
            checkpoint.task_description,
            resume_from=task_id,
        )

        # Display result
        status_color = {"completed": "green", "failed": "red"}.get(
            result.status.value, "yellow"
        )
        console.print(f"\n[{status_color}]Status: {result.status.value}[/{status_color}]")

        if result.summary:
            console.print(f"\n{result.summary}")

        if result.error:
            console.print(f"\n[red]Error: {result.error}[/red]")

        # Display agent learning summary
        from modular_agents.reporting import AgentSummaryReporter

        module_map = runtime.get_module_map()
        AgentSummaryReporter.display_agent_summary(result, module_map)
        AgentSummaryReporter.display_detailed_agent_learning(result, module_map)

        # Save learning summary
        if result.subtask_results:
            summary_path = repo_path / ".modular-agents" / f"summary_{result.task_id}.md"
            AgentSummaryReporter.save_learning_summary(result, str(summary_path), module_map)

    asyncio.run(do_resume())


@main.group()
def checkpoints():
    """Manage task checkpoints."""
    pass


@checkpoints.command("list")
@click.option("--path", type=click.Path(exists=True), default=".", help="Repository path")
@click.option("--task", type=str, help="Filter by task ID")
def list_checkpoints(path: str, task: str):
    """List all checkpoints.

    Shows available checkpoints for tasks with phase information.
    """
    from modular_agents.checkpoint import CheckpointManager
    from rich.table import Table

    repo_path = Path(path).resolve()
    checkpoint_manager = CheckpointManager()

    if task:
        # List checkpoints for specific task
        checkpoints = checkpoint_manager.list_checkpoints(task)

        if not checkpoints:
            console.print(f"[yellow]No checkpoints found for task: {task}[/yellow]")
            return

        console.print(f"[bold]Checkpoints for task:[/bold] {task}\n")

        table = Table()
        table.add_column("Checkpoint ID", style="cyan")
        table.add_column("Phase", justify="right", style="yellow")
        table.add_column("Timestamp", style="green")
        table.add_column("Completed", justify="right", style="blue")
        table.add_column("Pending", justify="right", style="magenta")

        for cp in checkpoints:
            from datetime import datetime
            timestamp = datetime.fromisoformat(cp["timestamp"])
            table.add_row(
                cp["checkpoint_id"][:8],
                f"{cp['phase_index'] + 1}/{cp['total_phases']}",
                timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                str(cp["completed_count"]),
                str(cp["pending_count"]),
            )

        console.print(table)
    else:
        # List all tasks with checkpoints
        checkpoint_dir = repo_path / ".modular-agents" / "checkpoints"

        if not checkpoint_dir.exists():
            console.print("[yellow]No checkpoints directory found[/yellow]")
            return

        task_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir()]

        if not task_dirs:
            console.print("[yellow]No checkpoints found[/yellow]")
            return

        console.print(f"[bold]Tasks with checkpoints:[/bold] {len(task_dirs)}\n")

        table = Table()
        table.add_column("Task ID", style="cyan")
        table.add_column("Checkpoints", justify="right", style="yellow")
        table.add_column("Latest Phase", style="green")

        for task_dir in task_dirs:
            task_id = task_dir.name
            checkpoints = checkpoint_manager.list_checkpoints(task_id)

            if checkpoints:
                latest = checkpoints[-1]
                table.add_row(
                    task_id[:16],
                    str(len(checkpoints)),
                    f"{latest['phase_index'] + 1}/{latest['total_phases']}",
                )

        console.print(table)
        console.print(f"\n[dim]Use --task <id> to see details for a specific task[/dim]")


@checkpoints.command("clean")
@click.option("--path", type=click.Path(exists=True), default=".", help="Repository path")
@click.option("--older-than", type=int, default=7, help="Delete checkpoints older than N days")
@click.option("--task", type=str, help="Clean specific task (otherwise clean all)")
def clean_checkpoints(path: str, older_than: int, task: str):
    """Clean old checkpoints.

    Removes checkpoints older than the specified number of days.
    """
    from modular_agents.checkpoint import CheckpointManager
    from rich.prompt import Confirm

    repo_path = Path(path).resolve()
    checkpoint_manager = CheckpointManager()

    console.print(f"[bold]Cleaning checkpoints older than {older_than} days...[/bold]\n")

    if task:
        console.print(f"[dim]Task filter: {task}[/dim]")

    deleted = checkpoint_manager.cleanup_old_checkpoints(days=older_than)

    if deleted > 0:
        console.print(f"[green]✓[/green] Deleted {deleted} old checkpoint(s)")
    else:
        console.print("[dim]No old checkpoints found[/dim]")


@main.command()
@click.argument("task_id", required=True)
@click.option("--path", type=click.Path(exists=True), default=".", help="Repository path")
@click.option("--watch", "-w", is_flag=True, help="Live update mode (refresh every 2 seconds)")
def progress(task_id: str, path: str, watch: bool):
    """Show task progress.

    TASK_ID is the ID of the task to monitor.
    """
    from modular_agents.progress import ProgressTracker
    import time

    repo_path = Path(path).resolve()
    tracker = ProgressTracker()

    # Load progress
    progress_state = tracker.load_progress(task_id)

    if not progress_state:
        console.print(f"[red]No progress found for task: {task_id}[/red]")
        console.print(f"Progress files are stored in: {repo_path / '.modular-agents' / 'progress'}")
        return

    if watch:
        # Live updating mode
        console.print("[dim]Press Ctrl+C to exit[/dim]\n")

        try:
            while True:
                # Clear screen (works on most terminals)
                import os
                os.system('clear' if os.name != 'nt' else 'cls')

                # Reload progress
                progress_state = tracker.load_progress(task_id)

                if progress_state:
                    tracker.display_progress(live=False)  # Display once

                    # Check if task is complete
                    if progress_state.status.value in ["completed", "failed"]:
                        console.print(f"\n[bold]Task {progress_state.status.value}![/bold]")
                        break
                else:
                    console.print(f"[yellow]Progress file removed or task completed[/yellow]")
                    break

                # Wait before refresh
                time.sleep(2)

        except KeyboardInterrupt:
            console.print("\n[dim]Stopped monitoring[/dim]")

    else:
        # Single display
        tracker.display_progress(live=False)


@main.command()
@click.argument("path", type=click.Path(exists=True), default=".")
def enrich(path: str):
    """Enrich existing knowledge with custom information.

    PATH is the repository (default: current directory).

    This command allows you to add custom metadata, descriptions, and notes
    to existing repository knowledge without re-analyzing.
    """
    from modular_agents.knowledge_manager import KnowledgeManager

    repo_path = Path(path).resolve()

    # Load existing knowledge
    knowledge = KnowledgeManager.load_knowledge(repo_path)
    if not knowledge:
        console.print("[red]No knowledge found. Run 'anton init' first.[/red]")
        return

    # Display current knowledge
    KnowledgeManager.display_knowledge_summary(knowledge)

    # Interactive enrichment
    KnowledgeManager.interactive_enrich(knowledge, repo_path)


@main.command()
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def export(path: str, output: str):
    """Export knowledge to markdown.

    PATH is the repository (default: current directory).
    """
    from modular_agents.knowledge_manager import KnowledgeManager

    repo_path = Path(path).resolve()

    # Load knowledge
    knowledge = KnowledgeManager.load_from_hierarchy(repo_path)
    if not knowledge:
        console.print("[red]No knowledge found. Run 'anton init' first.[/red]")
        return

    # Enrich with custom metadata
    knowledge = KnowledgeManager.enrich_knowledge(knowledge, repo_path)

    # Determine output path
    if output:
        output_path = Path(output)
    else:
        output_path = repo_path / ".modular-agents" / "knowledge.md"

    # Export
    KnowledgeManager.export_knowledge(knowledge, output_path)


@main.command()
@click.option("--db", "-db", type=click.Path(), help="Database path (default: ~/.modular-agents/knowledge.db)")
@click.option("--model", "-m", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model to use")
@click.option("--device", "-d", default="cpu", help="Device for embeddings (cpu, cuda, mps)")
def setup(db: str, model: str, device: str):
    """Set up the global knowledge base.

    Initializes the SQLite database with vector search capabilities.
    """
    from modular_agents.knowledge.embeddings import GemmaEmbeddingProvider
    from modular_agents.knowledge.store import KnowledgeStore

    # Determine database path
    if db:
        db_path = Path(db)
    else:
        db_path = Path.home() / ".modular-agents" / "knowledge.db"

    console.print(f"[bold]Setting up knowledge base:[/bold] {db_path}")
    console.print(f"[dim]Embedding model: {model}[/dim]")
    console.print(f"[dim]Device: {device}[/dim]\n")

    # Create embedding provider
    try:
        embedding_provider = GemmaEmbeddingProvider(
            model_name=model,
            device=device,
        )
        console.print(f"[green]✓[/green] Embedding provider initialized")
    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("\nInstall dependencies with:")
        console.print("  pip install transformers torch sentencepiece")
        return

    # Create knowledge store
    store = KnowledgeStore(db_path, embedding_provider)

    try:
        store.connect()
        console.print(f"[green]✓[/green] Database connected")

        # Show stats
        stats = store.get_stats()
        console.print(f"\n[bold]Knowledge Base Statistics:[/bold]")
        console.print(f"  Repositories: {stats['total_repos']}")
        console.print(f"  Code chunks: {stats['total_chunks']}")
        console.print(f"  Embeddings: {stats['total_embeddings']}")
        console.print(f"  Relations: {stats['total_relations']}")
        console.print(f"  Task learnings: {stats['total_learnings']}")

        console.print(f"\n[green]✓[/green] Knowledge base ready!")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        store.close()


@main.command()
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--db", "-db", type=click.Path(), help="Database path (default: ~/.modular-agents/knowledge.db)")
@click.option("--provider", "-p", type=click.Choice(["claude", "openai", "ollama"]), default="claude", help="LLM provider")
@click.option("--model", "-m", default=None, help="LLM model name")
@click.option("--api-key", "-k", envvar="LLM_API_KEY", default=None, help="API key")
@click.option("--base-url", "-u", default=None, help="Base URL for API (for local servers)")
@click.option("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model")
@click.option("--device", default="cpu", help="Device for embeddings")
@click.option("--force", "-f", is_flag=True, help="Re-index even if already indexed")
@click.option("--no-embeddings", is_flag=True, help="Skip generating embeddings (faster)")
@click.option("--simple-parser", is_flag=True, help="Use regex-based parser instead of LLM (no API needed)")
@click.option("--debug", is_flag=True, help="Show detailed error messages")
def index(
    path: str,
    db: str,
    provider: str,
    model: str,
    api_key: str,
    base_url: str,
    embedding_model: str,
    device: str,
    force: bool,
    no_embeddings: bool,
    simple_parser: bool,
    debug: bool,
):
    """Index a repository into the knowledge base.

    PATH is the repository to index (default: current directory).

    This command analyzes code files, breaks them into chunks using LLM,
    generates embeddings, and stores everything in the knowledge base.
    """
    from modular_agents.knowledge import (
        GemmaEmbeddingProvider,
        KnowledgeStore,
        LLMCodeParser,
        RepositoryIndexer,
    )
    from modular_agents.knowledge_manager import KnowledgeManager
    from modular_agents.llm import LLMConfig, LLMProviderRegistry

    repo_path = Path(path).resolve()

    # Load repository knowledge
    console.print(f"[bold]Loading repository knowledge...[/bold]")
    repo_knowledge = KnowledgeManager.load_from_hierarchy(repo_path)

    if not repo_knowledge:
        console.print(
            f"[red]No repository knowledge found. Run 'anton init {path}' first.[/red]"
        )
        return

    console.print(f"[green]✓[/green] Loaded knowledge for {len(repo_knowledge.modules)} modules")

    # Determine database path
    if db:
        db_path = Path(db)
    else:
        db_path = Path.home() / ".modular-agents" / "knowledge.db"

    if not db_path.exists():
        console.print(f"[red]Knowledge base not found at {db_path}[/red]")
        console.print("Run 'anton setup' first to initialize the knowledge base.")
        return

    # Set up LLM
    default_models = {
        "claude": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
        "ollama": "llama3.1",
    }
    model = model or default_models.get(provider)

    if not api_key and provider in ("claude", "openai"):
        env_vars = {"claude": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
        api_key = os.environ.get(env_vars.get(provider, ""))

        if not api_key:
            # If using custom base_url (local server), use dummy key
            if base_url:
                api_key = "dummy-key-for-local-server"
                console.print(f"[yellow]Using dummy API key for local server[/yellow]")
            else:
                console.print(f"[red]API key required for {provider}[/red]")
                console.print(f"Set {env_vars.get(provider)} or use --api-key")
                return

    llm_config = LLMConfig(model=model, api_key=api_key, base_url=base_url)
    llm = LLMProviderRegistry.create(provider, llm_config)

    console.print(f"[green]✓[/green] LLM provider: {provider} ({model})")

    # Set up embedding provider
    embedding_provider = None
    if not no_embeddings:
        try:
            embedding_provider = GemmaEmbeddingProvider(
                model_name=embedding_model,
                device=device,
            )
            console.print(f"[green]✓[/green] Embedding provider initialized")
        except ImportError:
            console.print(
                "[yellow]Warning: Could not load embedding provider. "
                "Indexing without embeddings.[/yellow]"
            )
            console.print("Install with: pip install transformers torch sentencepiece")

    # Create components
    store = KnowledgeStore(db_path, embedding_provider)

    # Choose parser based on --simple-parser flag
    if simple_parser:
        from modular_agents.knowledge.simple_parser import SimpleCodeParser
        parser = SimpleCodeParser()
        console.print("[yellow]Using simple regex-based parser (no LLM)[/yellow]")
    else:
        parser = LLMCodeParser(llm)
        console.print(f"[green]✓[/green] Using LLM-assisted parser")

    indexer = RepositoryIndexer(store, parser, embedding_provider)

    # Index repository
    async def do_index():
        try:
            store.connect()
            stats = await indexer.index_repository(
                repo_knowledge,
                force=force,
                show_progress=True,
            )

            # Display results
            if stats.get("up_to_date"):
                console.print(f"\n[bold green]Repository Up to Date![/bold green]")
                console.print(f"  Skipped files: {stats['skipped_files']}")
                if stats.get("deleted_files", 0) > 0:
                    console.print(f"  Deleted files: {stats['deleted_files']}")
            else:
                console.print(f"\n[bold green]Indexing Complete![/bold green]")
                console.print(f"  Files indexed: {stats['total_files']}")
                console.print(f"  Chunks created: {stats['total_chunks']}")

                # Show incremental stats
                if stats.get("skipped_files", 0) > 0:
                    console.print(f"  [green]Skipped unchanged: {stats['skipped_files']}[/green]")
                if stats.get("deleted_files", 0) > 0:
                    console.print(f"  [yellow]Deleted files: {stats['deleted_files']}[/yellow]")

                if stats.get("failed_files", 0) > 0:
                    console.print(
                        f"  [red]Failed files: {stats['failed_files']}[/red]"
                    )

                    # Show error details
                    if stats.get("errors"):
                        console.print(f"\n[red]Errors:[/red]")
                        error_limit = None if debug else 10
                        for error in stats["errors"][:error_limit]:
                            console.print(f"  {error}")

            # Show updated stats
            db_stats = store.get_stats()
            console.print(f"\n[bold]Knowledge Base Total:[/bold]")
            console.print(f"  Repositories: {db_stats['total_repos']}")
            console.print(f"  Code chunks: {db_stats['total_chunks']}")
            console.print(f"  Embeddings: {db_stats['total_embeddings']}")

        finally:
            store.close()

    asyncio.run(do_index())


@main.command()
@click.argument("query", required=True)
@click.option("--db", "-db", type=click.Path(), help="Database path")
@click.option("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model")
@click.option("--device", default="cpu", help="Device for embeddings")
@click.option("--limit", "-n", default=10, help="Number of results")
@click.option("--repo", "-r", help="Filter by repository path")
@click.option("--module", "-mod", help="Filter by module name")
def knowledge(query: str, db: str, embedding_model: str, device: str, limit: int, repo: str, module: str):
    """Search the knowledge base for similar code.

    QUERY is the search query (e.g., "user authentication logic").
    """
    from modular_agents.knowledge.embeddings import GemmaEmbeddingProvider
    from modular_agents.knowledge.store import KnowledgeStore
    from rich.syntax import Syntax
    from rich.panel import Panel

    # Determine database path
    if db:
        db_path = Path(db)
    else:
        db_path = Path.home() / ".modular-agents" / "knowledge.db"

    if not db_path.exists():
        console.print(f"[red]Knowledge base not found at {db_path}[/red]")
        console.print("Run 'anton setup' first to initialize the knowledge base.")
        return

    console.print(f"[bold]Searching for:[/bold] {query}\n")

    # Create components
    embedding_provider = GemmaEmbeddingProvider(
        model_name=embedding_model,
        device=device,
    )
    store = KnowledgeStore(db_path, embedding_provider)

    async def do_search():
        try:
            store.connect()

            # Search
            results = await store.search_similar(
                query=query,
                limit=limit,
                repo_path=repo,
                module_name=module,
            )

            if not results:
                console.print("[yellow]No results found[/yellow]")
                return

            # Display results
            console.print(f"[bold]Found {len(results)} results:[/bold]\n")

            for i, (chunk, score) in enumerate(results, 1):
                # Create syntax-highlighted code
                syntax = Syntax(
                    chunk.content,
                    chunk.language,
                    theme="monokai",
                    line_numbers=True,
                    start_line=chunk.start_line,
                )

                # Create panel with metadata
                title = f"[{i}] {chunk.name} ({chunk.chunk_type}) - Score: {score:.3f}"
                subtitle = f"{chunk.module_name} | {chunk.file_path}:{chunk.start_line}-{chunk.end_line}"

                panel = Panel(
                    syntax,
                    title=title,
                    subtitle=subtitle,
                    border_style="blue",
                )

                console.print(panel)

                if chunk.summary:
                    console.print(f"[dim]Summary: {chunk.summary}[/dim]")
                if chunk.purpose:
                    console.print(f"[dim]Purpose: {chunk.purpose}[/dim]")
                console.print()

        finally:
            store.close()

    asyncio.run(do_search())


@main.command()
@click.option("--db", "-db", type=click.Path(), help="Database path (default: ~/.modular-agents/knowledge.db)")
@click.option("--json", is_flag=True, help="Output as JSON")
def list_projects(db: str, json: bool):
    """List all indexed projects in the knowledge base.

    Shows project paths, languages, chunk counts, and last index datetime.
    """
    from modular_agents.knowledge import KnowledgeStore

    # Determine database path
    if db:
        db_path = Path(db)
    else:
        db_path = Path.home() / ".modular-agents" / "knowledge.db"

    if not db_path.exists():
        console.print(f"[red]Knowledge base not found at {db_path}[/red]")
        console.print("Run 'anton index <path>' first to create the knowledge base.")
        return

    # Connect to knowledge store (no embedding provider needed for listing)
    store = KnowledgeStore(db_path, embedding_provider=None)

    try:
        store.connect()

        # Get all repos
        repos = store.list_repos()

        if not repos:
            console.print("[yellow]No indexed projects found[/yellow]")
            return

        # JSON output
        if json:
            import json as json_lib
            output = [
                {
                    "repo_path": repo.repo_path,
                    "project_type": repo.project_type.value,
                    "language": repo.language,
                    "total_chunks": repo.total_chunks,
                    "total_files": repo.total_files,
                    "indexed_at": repo.indexed_at.isoformat(),
                    "last_updated": repo.last_updated.isoformat(),
                }
                for repo in repos
            ]
            console.print(json_lib.dumps(output, indent=2))
            return

        # Rich table output
        from rich.table import Table

        table = Table(title=f"Indexed Projects ({len(repos)} total)")
        table.add_column("Project Path", style="cyan", no_wrap=False)
        table.add_column("Type", style="green")
        table.add_column("Language", style="blue")
        table.add_column("Files", justify="right", style="yellow")
        table.add_column("Chunks", justify="right", style="magenta")
        table.add_column("Last Indexed", style="dim")

        for repo in repos:
            # Format datetime as relative time
            from datetime import datetime
            now = datetime.now()
            delta = now - repo.last_updated

            if delta.days == 0:
                if delta.seconds < 3600:
                    time_ago = f"{delta.seconds // 60}m ago"
                else:
                    time_ago = f"{delta.seconds // 3600}h ago"
            elif delta.days == 1:
                time_ago = "1 day ago"
            elif delta.days < 7:
                time_ago = f"{delta.days} days ago"
            elif delta.days < 30:
                time_ago = f"{delta.days // 7} weeks ago"
            elif delta.days < 365:
                time_ago = f"{delta.days // 30} months ago"
            else:
                time_ago = f"{delta.days // 365} years ago"

            table.add_row(
                repo.repo_path,
                repo.project_type.value,
                repo.language,
                str(repo.total_files),
                str(repo.total_chunks),
                time_ago,
            )

        console.print(table)

        # Show stats
        total_files = sum(r.total_files for r in repos)
        total_chunks = sum(r.total_chunks for r in repos)
        console.print(
            f"\n[dim]Total: {total_files} files, {total_chunks} chunks across {len(repos)} projects[/dim]"
        )

    finally:
        store.close()


@main.command()
def providers():
    """List available LLM providers."""
    available = LLMProviderRegistry.available()

    console.print("[bold]Available LLM Providers:[/bold]\n")

    all_providers = ["claude", "openai", "ollama"]
    for p in all_providers:
        if p in available:
            console.print(f"  [green]✓[/green] {p}")
        else:
            console.print(f"  [dim]✗ {p} (not installed)[/dim]")

    console.print("\nInstall providers with:")
    console.print("  pip install modular-agents[claude]")
    console.print("  pip install modular-agents[openai]")
    console.print("  pip install modular-agents[ollama]")
    console.print("  pip install modular-agents[all]")


if __name__ == "__main__":
    main()
