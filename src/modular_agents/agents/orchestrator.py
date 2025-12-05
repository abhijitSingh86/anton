"""Orchestrator agent - coordinates all module agents."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import TYPE_CHECKING

from modular_agents.core.models import (
    AgentMessage,
    ExecutionPhase,
    ExecutionPlan,
    MessageType,
    RepoKnowledge,
    SubTask,
    SubTaskResult,
    Task,
    TaskResult,
    TaskStatus,
)
from modular_agents.llm import LLMProvider

from .base import BaseAgent
from .module_agent import ModuleAgent
from .prompts import build_orchestrator_prompt

if TYPE_CHECKING:
    from modular_agents.autonomy import AutonomyConfig, AutonomyManager
    from modular_agents.checkpoint import AgentState, CheckpointManager
    from modular_agents.knowledge.store import KnowledgeStore
    from modular_agents.progress import ProgressTracker
    from modular_agents.retry import RetryConfig, RetryManager
    from modular_agents.tools.base import ToolRegistry


class OrchestratorAgent(BaseAgent):
    """Orchestrator that coordinates module agents.
    
    Responsibilities:
    - Receive high-level tasks from users
    - Decompose tasks into module-specific subtasks
    - Delegate to appropriate module agents
    - Coordinate cross-module work
    - Integrate results
    """
    
    def __init__(
        self,
        repo_knowledge: RepoKnowledge,
        module_agents: dict[str, ModuleAgent],
        llm: LLMProvider,
        knowledge_store: KnowledgeStore | None = None,
        tool_registry: ToolRegistry | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        retry_manager: RetryManager | None = None,
        progress_tracker: ProgressTracker | None = None,
        autonomy_config: AutonomyConfig | None = None,
        autonomy_manager: AutonomyManager | None = None,
    ):
        self.repo_knowledge = repo_knowledge
        self.module_agents = module_agents
        self.knowledge_store = knowledge_store
        self.tool_registry = tool_registry

        # Continuation system components
        self.checkpoint_manager = checkpoint_manager
        self.retry_manager = retry_manager
        self.progress_tracker = progress_tracker
        self.autonomy_config = autonomy_config
        self.autonomy_manager = autonomy_manager

        system_prompt = build_orchestrator_prompt(repo_knowledge)

        super().__init__(
            name="orchestrator",
            llm=llm,
            system_prompt=system_prompt,
        )
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process incoming message."""
        if message.message_type == MessageType.TASK_ASSIGNMENT:
            task_desc = message.payload.get("description", "")
            result = await self.process_task(task_desc)
            return AgentMessage(
                id=str(uuid.uuid4()),
                sender=self.name,
                recipient=message.sender,
                message_type=MessageType.TASK_RESULT,
                payload=result.model_dump(),
                correlation_id=message.correlation_id,
            )
        
        return AgentMessage(
            id=str(uuid.uuid4()),
            sender=self.name,
            recipient=message.sender,
            message_type=MessageType.RESPONSE,
            payload={"status": "unknown message type"},
            correlation_id=message.correlation_id,
        )
    
    async def process_task(
        self,
        task_description: str,
        resume_from: str | None = None,
    ) -> TaskResult:
        """Process a high-level task end-to-end.

        Args:
            task_description: Description of the task to perform
            resume_from: Optional task ID to resume from checkpoint

        Returns:
            Task result with status and outputs
        """
        from rich.console import Console
        console = Console()

        # Try to resume from checkpoint if specified
        if resume_from and self.checkpoint_manager:
            checkpoint = self.checkpoint_manager.load_checkpoint(resume_from)
            if checkpoint:
                console.print(f"[cyan]Resuming from checkpoint: {checkpoint.checkpoint_id}[/cyan]")
                return await self._resume_from_checkpoint(checkpoint)
            else:
                console.print(f"[yellow]Warning: Checkpoint not found for task {resume_from}[/yellow]")
                console.print("[yellow]Starting fresh execution...[/yellow]")

        # Normal execution path
        task_id = str(uuid.uuid4())[:8]

        # Initialize progress tracking
        if self.progress_tracker:
            self.progress_tracker.start_task(task_id, task_description)

        # Step 1: Decompose the task
        subtasks = await self.decompose_task(task_id, task_description)

        if not subtasks:
            error_msg = (
                "Could not decompose task into subtasks. "
                "The orchestrator failed to generate valid subtasks from the task description. "
                "\n\nPossible causes:"
                "\n1. The task description is too vague or unclear"
                "\n2. No modules match the task requirements"
                "\n3. The LLM failed to return valid JSON"
                "\n\nSuggestions:"
                "\n- Try rephrasing the task more specifically"
                "\n- Verify modules exist: anton analyze ."
                "\n- Run with --debug to see full LLM response"
            )

            console.print(f"\n[bold red]Task Decomposition Failed[/bold red]")
            console.print(error_msg)

            if self.progress_tracker:
                self.progress_tracker.task_completed(TaskStatus.FAILED)

            return TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error="Could not decompose task into subtasks (see above for details)",
            )

        # Build module mapping for reporting
        self.module_map = {st.id: st.module for st in subtasks}

        # Step 2: Create execution plan
        plan = await self.create_execution_plan(task_id, subtasks)

        # Update progress tracker with plan
        if self.progress_tracker:
            self.progress_tracker.set_plan(plan)

        # Step 3: Execute phases with checkpointing
        all_results: list[SubTaskResult] = []

        for phase_idx, phase in enumerate(plan.phases):
            # Mark phase as started
            if self.progress_tracker:
                self.progress_tracker.phase_started(phase_idx)

            # Execute phase with retry support
            phase_results = await self._execute_phase_with_continuation(
                phase,
                subtasks,
                phase_idx,
            )
            all_results.extend(phase_results)

            # Save checkpoint after each phase
            if self.checkpoint_manager and self.autonomy_config and self.autonomy_config.auto_checkpoint:
                await self._create_and_save_checkpoint(
                    task_id,
                    task_description,
                    phase_idx,
                    len(plan.phases),
                    all_results,
                    subtasks,
                    plan,
                )

            # Mark phase as completed
            if self.progress_tracker:
                self.progress_tracker.phase_completed(phase_idx)

            # Check for failures
            failed = [r for r in phase_results if r.status == TaskStatus.FAILED]
            if failed and not (self.autonomy_config and self.autonomy_config.auto_retry):
                # Fail immediately if no auto-retry
                if self.progress_tracker:
                    self.progress_tracker.task_completed(TaskStatus.FAILED)

                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    subtask_results=all_results,
                    error=f"Phase {phase.phase_number} failed: {failed[0].error}",
                )

            # Check for blockers
            blocked = [r for r in phase_results if r.status == TaskStatus.BLOCKED]
            if blocked:
                blockers = []
                for r in blocked:
                    blockers.extend(r.blockers)

                if self.progress_tracker:
                    self.progress_tracker.task_completed(TaskStatus.BLOCKED)

                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.BLOCKED,
                    subtask_results=all_results,
                    error=f"Blocked: {', '.join(blockers)}",
                )

        # Step 4: Generate summary
        summary = await self._generate_summary(task_description, all_results)

        # Mark task as completed
        if self.progress_tracker:
            self.progress_tracker.task_completed(TaskStatus.COMPLETED)

        return TaskResult(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            subtask_results=all_results,
            summary=summary,
        )
    
    async def _get_relevant_code_context(self, task_description: str) -> str:
        """Query knowledge base for code relevant to the task.

        Args:
            task_description: The task description to search for

        Returns:
            Formatted string with relevant code snippets, or empty if no knowledge store
        """
        if not self.knowledge_store:
            return ""

        try:
            # Search for relevant code chunks
            results = await self.knowledge_store.search_similar(
                query=task_description,
                limit=5,
                repo_path=str(self.repo_knowledge.root_path),
            )

            if not results:
                return ""

            # Format the results
            context_parts = ["\n## Relevant Code Context from Knowledge Base:\n"]

            for chunk, similarity in results:
                context_parts.append(f"\n### {chunk.file_path} - {chunk.name} ({chunk.chunk_type})")
                context_parts.append(f"Similarity: {similarity:.2f}")
                if chunk.summary:
                    context_parts.append(f"Summary: {chunk.summary}")
                context_parts.append(f"```{chunk.language}")
                # Limit code snippet size
                code_preview = chunk.content[:500]
                if len(chunk.content) > 500:
                    code_preview += "\n... (truncated)"
                context_parts.append(code_preview)
                context_parts.append("```\n")

            return "\n".join(context_parts)

        except Exception as e:
            # Don't fail the task if knowledge base query fails
            from rich.console import Console
            console = Console()
            console.print(f"[yellow]Warning: Knowledge base query failed: {e}[/yellow]")
            return ""

    async def _explore_codebase_for_task(self, task_description: str) -> str:
        """Actively explore codebase using tools to understand existing structure.

        Args:
            task_description: The task to explore for

        Returns:
            Formatted string with discovered code structure
        """
        if not self.tool_registry:
            return ""

        from rich.console import Console
        console = Console()
        console.print("[dim]🔍 Exploring codebase...[/dim]")

        # Use LLM with tool calling to explore the codebase
        exploration_prompt = f"""Before decomposing this task, explore the codebase to understand what already exists:

Task: {task_description}

Use available tools to:
1. Find existing files related to this task (use grep_codebase)
2. Check what models/classes already exist (use grep_codebase for "case class", "class", "trait")
3. Identify existing API endpoints (use grep_codebase for "Method.GET", "Method.POST", etc.)
4. Read relevant files to understand the architecture (use read_file)

After exploring, respond with a summary in this format:
```
DISCOVERED:
- Existing models: [list models found]
- Existing endpoints: [list API endpoints found]
- Relevant files: [list file paths]
- Architecture notes: [observations about the codebase]
```

Focus on finding what ALREADY EXISTS so we don't duplicate or contradict it."""

        try:
            # Get tools in appropriate format
            tools = self.tool_registry.to_anthropic_format()
            if self.llm.name == "openai":
                tools = self.tool_registry.to_openai_format()

            # Build messages
            messages = [{"role": "user", "content": exploration_prompt}]

            # Tool calling loop
            max_rounds = 5
            tool_calls_made = False

            for round_num in range(max_rounds):
                # Call LLM with tools
                response_text, tool_calls = await self.llm.chat_with_tools(
                    messages=messages,
                    system=self.system_prompt,
                    tools=tools,
                    max_tool_rounds=1,
                )

                # If no tool calls, we have final response
                if not tool_calls:
                    # Check if ANY tools were called during exploration
                    if not tool_calls_made and round_num == 0:
                        console.print(f"[yellow]⚠ LLM did not use tools - may not support tool calling[/yellow]")
                        console.print(f"[yellow]  Skipping exploration (results may be hallucinated)[/yellow]")
                        return ""  # Return empty to avoid using hallucinated data

                    console.print(f"[dim]✓ Exploration complete ({round_num + 1} tool rounds)[/dim]")
                    return f"\n## Codebase Exploration Results:\n{response_text}\n"

                # Mark that we got tool calls
                tool_calls_made = True

                # Execute tool calls
                for tool_call in tool_calls:
                    result = await self.tool_registry.execute(tool_call)

                    # Add assistant message with tool call
                    messages.append({
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": tool_call.id, "name": tool_call.name, "input": tool_call.parameters}] if self.llm.name == "claude" else f"Calling tool: {tool_call.name}"
                    })

                    # Add tool result as user message
                    if result.success:
                        messages.append({
                            "role": "user",
                            "content": f"Tool result from {tool_call.name}:\n{result.output}"
                        })
                    else:
                        messages.append({
                            "role": "user",
                            "content": f"Tool {tool_call.name} failed: {result.error}"
                        })

            # Max rounds exceeded - return what we have
            console.print(f"[yellow]⚠ Exploration hit max rounds[/yellow]")
            return "\n## Codebase Exploration: (partial results)\n"

        except Exception as e:
            console.print(f"[yellow]⚠ Exploration failed: {e}[/yellow]")
            return ""

    def _parse_exploration_results(self, exploration_text: str) -> dict:
        """Parse exploration results to extract structured information.

        Args:
            exploration_text: Raw exploration results from LLM

        Returns:
            Dict with discovered files, models, endpoints
        """
        import re

        discovered = {
            "files": [],
            "models": [],
            "endpoints": [],
            "notes": []
        }

        if not exploration_text:
            return discovered

        # Extract file paths (anything ending in .scala, .java, .py, etc.)
        file_pattern = r'[\w/\-\.]+\.(scala|java|py|ts|js|go|rs)'
        files = re.findall(file_pattern, exploration_text, re.IGNORECASE)
        discovered["files"] = list(set(files))  # Deduplicate

        # Extract model/class names (case class Foo, class Bar, trait Baz)
        model_patterns = [
            r'case class (\w+)',
            r'class (\w+)',
            r'trait (\w+)',
            r'interface (\w+)',
            r'type (\w+)',
        ]
        for pattern in model_patterns:
            models = re.findall(pattern, exploration_text)
            discovered["models"].extend(models)
        discovered["models"] = list(set(discovered["models"]))

        # Extract API endpoints (Method.GET /path, POST /path, etc.)
        endpoint_pattern = r'(GET|POST|PUT|DELETE|PATCH)\s+([/\w\-{}:]+)'
        endpoints = re.findall(endpoint_pattern, exploration_text, re.IGNORECASE)
        discovered["endpoints"] = [f"{method} {path}" for method, path in endpoints]

        # Extract architecture notes (lines containing "architecture", "pattern", "uses")
        for line in exploration_text.split('\n'):
            if any(keyword in line.lower() for keyword in ['architecture', 'pattern', 'uses', 'structure']):
                discovered["notes"].append(line.strip())

        return discovered

    async def decompose_task(
        self, task_id: str, description: str
    ) -> list[SubTask]:
        """Decompose a task into module-specific subtasks."""
        from rich.console import Console
        from rich.syntax import Syntax

        console = Console()

        # Actively explore codebase first (using tools)
        exploration_context = await self._explore_codebase_for_task(description)

        # Parse exploration to extract structured data
        discovered = self._parse_exploration_results(exploration_context)

        # Get relevant code context from knowledge base (semantic search)
        code_context = await self._get_relevant_code_context(description)

        modules_info = "\n".join([
            f"- {m.name}: {m.purpose}\n  Path: {m.path}  ⚠️ ALL files for this module MUST start with: {m.path}/"
            for m in self.repo_knowledge.modules
        ])

        # Build explicit constraints from discovered data
        discovered_files_str = ""
        if discovered["files"]:
            files_by_ext = {}
            for f in discovered["files"]:
                ext = f.split('.')[-1]
                if ext not in files_by_ext:
                    files_by_ext[ext] = []
                files_by_ext[ext].append(f)

            discovered_files_str = "\n## DISCOVERED FILES (USE THESE EXACT PATHS):\n"
            for ext, files in files_by_ext.items():
                discovered_files_str += f"\n{ext.upper()} files:\n"
                for f in files[:10]:  # Limit to 10 per type
                    discovered_files_str += f"  - {f}\n"

        discovered_models_str = ""
        if discovered["models"]:
            discovered_models_str = f"\n## EXISTING MODELS/CLASSES (DO NOT RECREATE):\n"
            for model in discovered["models"][:20]:  # Limit to 20
                discovered_models_str += f"  - {model}\n"

        discovered_endpoints_str = ""
        if discovered["endpoints"]:
            discovered_endpoints_str = f"\n## EXISTING API ENDPOINTS:\n"
            for endpoint in discovered["endpoints"][:15]:  # Limit to 15
                discovered_endpoints_str += f"  - {endpoint}\n"

        prompt = f"""Decompose this task into module-specific subtasks:

Task: {description}

Available modules:
{modules_info}

Module dependencies:
{json.dumps(self.repo_knowledge.dependency_graph, indent=2)}
{discovered_files_str}
{discovered_models_str}
{discovered_endpoints_str}
{code_context}

CRITICAL RULES - FAILURE TO FOLLOW WILL CAUSE TASK FAILURE:
1. ONLY use file paths from "DISCOVERED FILES" section above
2. DO NOT create placeholder file names like "YourApiEndpoint.scala" or "User.scala"
3. Work with EXISTING models from "EXISTING MODELS/CLASSES" section
4. DO NOT duplicate existing models - modify or extend them instead
5. Each subtask must be scoped to a single module
6. Consider module dependencies for ordering
7. You MUST respond with valid JSON only, no additional text
8. 🚨 In subtask descriptions, INCLUDE the module's absolute path (shown above in "Available modules")
9. 🚨 In affected_files, use FULL ABSOLUTE PATHS starting with the module's path
10. 🚨 PROVIDE DETAILED CONTEXT in descriptions - what methods, fields, endpoints to create

Example of GOOD subtask description:
  "description": "Create UserRepository trait in /path/to/database/UserRepository.scala with methods: create(user: User): Task[User], findById(id: UUID): Task[Option[User]], findAll: Task[List[User]], update(id: UUID, user: User): Task[Option[User]], delete(id: UUID): Task[Boolean]. Follow existing repository patterns in the codebase."

Example of BAD subtask description:
  "description": "Create UserRepository.scala"  ← TOO VAGUE! Missing methods, path, and context

IMPORTANT: Subtask descriptions must include:
- WHAT to create/modify (specific class/trait name)
- WHERE (full absolute path)
- DETAILS (what methods, fields, endpoints, with signatures if applicable)
- CONTEXT (follow existing patterns, use similar code as reference)

If creating repositories, services, or APIs, specify the expected interface/methods based on similar existing code.

Respond with ONLY this JSON structure (no markdown, no explanations):
{{
  "subtasks": [
    {{
      "module": "module_name",
      "description": "Detailed description with WHAT, WHERE, DETAILS, and CONTEXT",
      "affected_files": ["FULL absolute paths from DISCOVERED FILES, e.g., /full/path/to/file.scala"],
      "depends_on": ["ids of subtasks that must complete first"]
    }}
  ]
}}

Use sequential IDs like "st_1", "st_2", etc. for depends_on references.
"""

        response = await self.think(prompt)

        # Try to extract JSON
        try:
            start = response.find('{')
            end = response.rfind('}') + 1

            if start < 0 or end <= start:
                # No JSON found
                console.print("[red]ERROR: No JSON found in orchestrator response[/red]")
                console.print("[yellow]Full response:[/yellow]")
                console.print(response[:1000])
                return []

            json_str = response[start:end]

            # Clean common issues
            import re
            json_str = re.sub(r',\s*}', '}', json_str)  # Trailing commas
            json_str = re.sub(r',\s*]', ']', json_str)

            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                console.print(f"[red]ERROR: JSON decode failed at line {e.lineno}, col {e.colno}[/red]")
                console.print(f"[red]Error: {e.msg}[/red]")
                console.print("\n[yellow]Attempted to parse:[/yellow]")
                syntax = Syntax(json_str[:500], "json", theme="monokai", line_numbers=True)
                console.print(syntax)
                console.print(f"\n[dim]Full response (first 1000 chars):[/dim]")
                console.print(response[:1000])
                return []

            # Validate structure
            if "subtasks" not in data:
                console.print("[red]ERROR: Response missing 'subtasks' field[/red]")
                console.print(f"[yellow]Got keys: {list(data.keys())}[/yellow]")
                console.print(f"[yellow]Data:[/yellow] {json.dumps(data, indent=2)[:500]}")
                return []

            if not isinstance(data["subtasks"], list):
                console.print("[red]ERROR: 'subtasks' is not a list[/red]")
                console.print(f"[yellow]Type: {type(data['subtasks'])}[/yellow]")
                return []

            if len(data["subtasks"]) == 0:
                console.print("[yellow]WARNING: Empty subtasks list[/yellow]")
                console.print("[dim]The orchestrator didn't generate any subtasks.[/dim]")
                return []

            # Build subtasks
            subtasks = []
            for i, st in enumerate(data["subtasks"]):
                # Validate required fields
                if "module" not in st or "description" not in st:
                    console.print(f"[red]ERROR: Subtask {i+1} missing required fields[/red]")
                    console.print(f"[yellow]Got: {st}[/yellow]")
                    continue

                subtask_id = f"{task_id}_st_{i+1}"

                # Map depends_on to actual IDs
                depends = []
                for dep in st.get("depends_on", []):
                    # Handle both "st_X" format and index references
                    if isinstance(dep, str) and dep.startswith("st_"):
                        idx = int(dep.split("_")[1]) - 1
                        if 0 <= idx < len(subtasks):
                            depends.append(subtasks[idx].id)
                    elif isinstance(dep, int):
                        # Direct index reference
                        if 0 <= dep - 1 < len(subtasks):
                            depends.append(subtasks[dep - 1].id)

                subtasks.append(SubTask(
                    id=subtask_id,
                    module=st["module"],
                    description=st["description"],
                    affected_files=st.get("affected_files", []),
                    depends_on=depends,
                    priority=i + 1,
                ))

            if subtasks:
                console.print(f"[green]✓[/green] Created {len(subtasks)} subtasks")
                for st in subtasks:
                    console.print(f"  - [{st.module}] {st.description[:60]}...")

            return subtasks

        except Exception as e:
            console.print(f"[red]ERROR: Unexpected error during decomposition: {type(e).__name__}[/red]")
            console.print(f"[red]{str(e)}[/red]")
            console.print(f"\n[dim]Response (first 1000 chars):[/dim]")
            console.print(response[:1000])
            import traceback
            console.print(f"\n[dim]Traceback:[/dim]")
            console.print(traceback.format_exc())
            return []
    
    async def create_execution_plan(
        self, task_id: str, subtasks: list[SubTask]
    ) -> ExecutionPlan:
        """Create an execution plan with parallelizable phases."""
        # Build dependency map
        subtask_map = {st.id: st for st in subtasks}
        
        # Topological sort into phases
        phases: list[ExecutionPhase] = []
        completed: set[str] = set()
        remaining = set(st.id for st in subtasks)
        
        phase_num = 1
        while remaining:
            # Find subtasks with all dependencies satisfied
            ready = []
            for st_id in remaining:
                st = subtask_map[st_id]
                if all(dep in completed for dep in st.depends_on):
                    ready.append(st_id)
            
            if not ready:
                # Circular dependency or error - just add remaining
                ready = list(remaining)
            
            phases.append(ExecutionPhase(
                phase_number=phase_num,
                subtask_ids=ready,
            ))
            
            completed.update(ready)
            remaining -= set(ready)
            phase_num += 1
        
        return ExecutionPlan(task_id=task_id, phases=phases)
    
    async def _execute_phase(
        self, phase: ExecutionPhase, subtasks: list[SubTask]
    ) -> list[SubTaskResult]:
        """Execute a phase of subtasks in parallel."""
        subtask_map = {st.id: st for st in subtasks}
        
        async def execute_subtask(subtask_id: str) -> SubTaskResult:
            subtask = subtask_map[subtask_id]
            module_name = subtask.module
            
            if module_name not in self.module_agents:
                return SubTaskResult(
                    subtask_id=subtask_id,
                    status=TaskStatus.FAILED,
                    error=f"No agent for module: {module_name}",
                )
            
            agent = self.module_agents[module_name]
            return await agent.execute_task(subtask)
        
        # Execute all subtasks in this phase concurrently
        results = await asyncio.gather(*[
            execute_subtask(st_id) for st_id in phase.subtask_ids
        ])
        
        return list(results)
    
    async def _generate_summary(
        self, task_description: str, results: list[SubTaskResult]
    ) -> str:
        """Generate a summary of the completed task."""
        changes_summary = []
        for r in results:
            if r.changes:
                changes_summary.append(f"- {r.subtask_id}: {len(r.changes)} file(s) changed")

        prompt = f"""Summarize the completed task:

Original task: {task_description}

Results:
{chr(10).join(changes_summary) if changes_summary else "No file changes"}

Provide a brief summary of what was accomplished.
"""

        return await self.think(prompt)

    # =========================================================================
    # Continuation System Methods
    # =========================================================================

    async def _resume_from_checkpoint(
        self,
        checkpoint: "Checkpoint",  # type: ignore
    ) -> TaskResult:
        """Resume task execution from a checkpoint.

        Args:
            checkpoint: Checkpoint to resume from

        Returns:
            Task result
        """
        from rich.console import Console
        console = Console()

        console.print(f"[bold cyan]Resuming Task Execution[/bold cyan]")
        console.print(f"Task: {checkpoint.task_description}")
        console.print(f"Resuming from phase {checkpoint.phase_index + 1}/{checkpoint.total_phases}")
        console.print(f"Completed subtasks: {len(checkpoint.completed_subtasks)}")
        console.print(f"Pending subtasks: {len(checkpoint.pending_subtasks)}\n")

        # Restore agent states
        for agent_name, state in checkpoint.agent_states.items():
            if agent_name in self.module_agents:
                agent = self.module_agents[agent_name]
                agent.restore_state(state)

        # Initialize progress tracking with checkpoint data
        if self.progress_tracker:
            # Load existing progress
            progress = self.progress_tracker.load_progress(checkpoint.task_id)
            if not progress:
                # Create new progress from checkpoint
                self.progress_tracker.start_task(
                    checkpoint.task_id,
                    checkpoint.task_description,
                    total_subtasks=len(checkpoint.completed_subtasks) + len(checkpoint.pending_subtasks),
                    total_phases=checkpoint.total_phases,
                )

        # Continue from next phase
        all_results = checkpoint.completed_subtasks.copy()

        # Build subtask map
        all_subtasks = checkpoint.completed_subtasks + checkpoint.pending_subtasks
        subtask_objects = []
        for st_result in all_subtasks:
            # Reconstruct SubTask from results (simplified)
            subtask = SubTask(
                id=st_result.subtask_id,
                module=checkpoint.metadata.get("subtask_modules", {}).get(st_result.subtask_id, "unknown"),
                description=checkpoint.metadata.get("subtask_descriptions", {}).get(st_result.subtask_id, ""),
                status=st_result.status,
            )
            subtask_objects.append(subtask)

        # Execute remaining phases
        for phase_idx in range(checkpoint.phase_index + 1, len(checkpoint.execution_plan.phases)):
            phase = checkpoint.execution_plan.phases[phase_idx]

            if self.progress_tracker:
                self.progress_tracker.phase_started(phase_idx)

            phase_results = await self._execute_phase_with_continuation(
                phase,
                subtask_objects,
                phase_idx,
            )
            all_results.extend([r for r in phase_results if r.subtask_id not in [cr.subtask_id for cr in all_results]])

            # Save new checkpoint
            if self.checkpoint_manager and self.autonomy_config and self.autonomy_config.auto_checkpoint:
                await self._create_and_save_checkpoint(
                    checkpoint.task_id,
                    checkpoint.task_description,
                    phase_idx,
                    checkpoint.total_phases,
                    all_results,
                    subtask_objects,
                    checkpoint.execution_plan,
                )

            if self.progress_tracker:
                self.progress_tracker.phase_completed(phase_idx)

            # Check for failures/blockers
            failed = [r for r in phase_results if r.status == TaskStatus.FAILED]
            if failed and not (self.autonomy_config and self.autonomy_config.auto_retry):
                if self.progress_tracker:
                    self.progress_tracker.task_completed(TaskStatus.FAILED)

                return TaskResult(
                    task_id=checkpoint.task_id,
                    status=TaskStatus.FAILED,
                    subtask_results=all_results,
                    error=f"Phase {phase.phase_number} failed after resume: {failed[0].error}",
                )

        # Generate summary
        summary = await self._generate_summary(checkpoint.task_description, all_results)

        if self.progress_tracker:
            self.progress_tracker.task_completed(TaskStatus.COMPLETED)

        console.print("\n[bold green]Task Resumed and Completed Successfully![/bold green]\n")

        return TaskResult(
            task_id=checkpoint.task_id,
            status=TaskStatus.COMPLETED,
            subtask_results=all_results,
            summary=summary,
        )

    async def _create_and_save_checkpoint(
        self,
        task_id: str,
        task_description: str,
        phase_index: int,
        total_phases: int,
        completed_subtasks: list[SubTaskResult],
        all_subtasks: list[SubTask],
        execution_plan: ExecutionPlan,
    ) -> None:
        """Create and save a checkpoint.

        Args:
            task_id: Task ID
            task_description: Task description
            phase_index: Current phase index
            total_phases: Total number of phases
            completed_subtasks: List of completed subtask results
            all_subtasks: All subtasks
            execution_plan: Execution plan
        """
        if not self.checkpoint_manager:
            return

        # Collect agent states
        agent_states = {}
        for agent_name, agent in self.module_agents.items():
            agent_states[agent_name] = agent.save_state()

        # Find pending subtasks
        completed_ids = {r.subtask_id for r in completed_subtasks}
        pending_subtasks = [st for st in all_subtasks if st.id not in completed_ids]

        # Create metadata
        metadata = {
            "subtask_modules": {st.id: st.module for st in all_subtasks},
            "subtask_descriptions": {st.id: st.description for st in all_subtasks},
        }

        # Create and save checkpoint
        checkpoint = self.checkpoint_manager.create_checkpoint(
            task_id=task_id,
            task_description=task_description,
            phase_index=phase_index,
            total_phases=total_phases,
            completed_subtasks=completed_subtasks,
            pending_subtasks=pending_subtasks,
            execution_plan=execution_plan,
            agent_states=agent_states,
            metadata=metadata,
        )

        # Update progress tracker with checkpoint path
        if self.progress_tracker:
            checkpoint_path = str(
                self.checkpoint_manager._get_checkpoint_file(task_id, checkpoint.checkpoint_id)
            )
            self.progress_tracker.update_checkpoint(checkpoint_path)

    async def _execute_phase_with_continuation(
        self,
        phase: ExecutionPhase,
        subtasks: list[SubTask],
        phase_idx: int,
    ) -> list[SubTaskResult]:
        """Execute a phase with retry and autonomy support.

        Args:
            phase: Execution phase to run
            subtasks: All subtasks
            phase_idx: Phase index

        Returns:
            List of subtask results
        """
        subtask_map = {st.id: st for st in subtasks}

        async def execute_subtask_with_retry(subtask_id: str) -> SubTaskResult:
            """Execute a single subtask with retry logic."""
            subtask = subtask_map[subtask_id]
            module_name = subtask.module

            if module_name not in self.module_agents:
                return SubTaskResult(
                    subtask_id=subtask_id,
                    status=TaskStatus.FAILED,
                    error=f"No agent for module: {module_name}",
                )

            agent = self.module_agents[module_name]

            # Track start time for timeout
            start_time = time.time()

            # Execute with retry if enabled
            if self.retry_manager and self.autonomy_config and self.autonomy_config.auto_retry:
                from modular_agents.retry import RetryConfig

                retry_config = RetryConfig(
                    max_retries=self.autonomy_config.max_autonomous_retries,
                )

                try:
                    result = await self.retry_manager.retry_with_backoff(
                        func=lambda: agent.execute_task(subtask),
                        config=retry_config,
                        context={"task_id": subtask_id, "subtask_id": subtask_id},
                    )
                except Exception as e:
                    # All retries exhausted
                    result = SubTaskResult(
                        subtask_id=subtask_id,
                        status=TaskStatus.FAILED,
                        error=f"Failed after {retry_config.max_retries} retries: {str(e)}",
                    )

                # Update progress tracker with retry count
                if self.progress_tracker and subtask_id in self.retry_manager.histories:
                    history = self.retry_manager.histories[subtask_id]
                    self.progress_tracker.subtask_retry(subtask_id, history.total_attempts)
            else:
                # No retry - single attempt
                result = await agent.execute_task(subtask)

            # Check timeout
            if self.autonomy_config:
                elapsed = time.time() - start_time
                if self.autonomy_manager and self.autonomy_manager.check_timeout(start_time, time.time()):
                    result.status = TaskStatus.FAILED
                    result.error = f"Subtask timed out after {elapsed:.1f}s"

            # Handle autonomy/approval workflow
            if self.autonomy_manager and result.status == TaskStatus.COMPLETED:
                # Validate safety checks
                is_safe, violations = self.autonomy_manager.validate_safety_checks(result.changes)

                if not is_safe:
                    # Critical violations
                    result.status = TaskStatus.BLOCKED
                    result.blockers = [v.message for v in violations if v.severity == "critical"]

                elif self.autonomy_config.level.value == "interactive":
                    # Always ask for approval
                    approved = await self._request_user_approval(result, violations)
                    if not approved:
                        result.status = TaskStatus.BLOCKED
                        result.blockers.append("User rejected changes")

                elif self.autonomy_config.level.value == "supervised":
                    # Ask for approval on risky changes
                    context = {
                        "file_count": len(result.changes),
                        "line_count": sum(len(c.content or "") for c in result.changes),
                    }
                    if self.autonomy_manager.should_request_approval("file_changes", context):
                        approved = await self._request_user_approval(result, violations)
                        if not approved:
                            result.status = TaskStatus.BLOCKED
                            result.blockers.append("User rejected changes")

            # Update progress tracker
            if self.progress_tracker:
                self.progress_tracker.subtask_completed(subtask_id, result)

            return result

        # Execute all subtasks in phase (in parallel if possible)
        results = await asyncio.gather(*[
            execute_subtask_with_retry(st_id) for st_id in phase.subtask_ids
        ])

        return list(results)

    async def _request_user_approval(
        self,
        result: SubTaskResult,
        violations: list,
    ) -> bool:
        """Request user approval for subtask changes.

        Args:
            result: Subtask result
            violations: List of safety violations

        Returns:
            True if approved, False otherwise
        """
        from rich.console import Console
        console = Console()

        if not self.autonomy_manager:
            return True  # No autonomy manager, default to approved

        # Generate approval prompt
        prompt = self.autonomy_manager.get_approval_prompt(result, violations)
        console.print(prompt)

        # Get user input
        response = input().strip().lower()
        return response in ["yes", "y"]
