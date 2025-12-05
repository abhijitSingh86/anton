"""Module agent - specialist for a single module."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from modular_agents.core.models import (
    AgentMessage,
    FileChange,
    MessageType,
    ModuleProfile,
    SubTask,
    SubTaskResult,
    TaskStatus,
)
from modular_agents.llm import LLMProvider

from .base import BaseAgent
from .prompts import build_module_agent_prompt


# =========================================================================
# Agent Scratchpad - Working Memory System
# =========================================================================

@dataclass
class AgentScratchpad:
    """Working memory for agent during task execution.

    This scratchpad helps agents keep track of:
    - What they've discovered
    - What they're planning to do
    - What they've created
    - What validations have passed/failed

    Prevents common errors like:
    - Forgetting context mid-task
    - Creating duplicate files
    - Violating path constraints
    - Losing track of decisions
    """

    # Task context
    task_id: str
    task_description: str
    module_path: str

    # Discovery phase
    discovered_files: List[str] = field(default_factory=list)
    discovered_patterns: List[Dict[str, str]] = field(default_factory=list)
    similar_code: List[str] = field(default_factory=list)
    knowledge_base_results: List[str] = field(default_factory=list)

    # Planning phase
    files_to_create: List[str] = field(default_factory=list)
    files_to_modify: List[str] = field(default_factory=list)
    dependencies_needed: List[str] = field(default_factory=list)
    implementation_approach: str = ""

    # Implementation phase
    created_files: List[str] = field(default_factory=list)
    modified_files: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)

    # Validation checklist
    path_validation_passed: bool = False
    language_validation_passed: bool = False
    placeholder_check_passed: bool = False
    tools_used: List[str] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        """Convert scratchpad to context string for LLM."""
        return f"""
╔════════════════════════════════════════════════════════════════╗
║                    🧠 YOUR WORKING MEMORY                      ║
╚════════════════════════════════════════════════════════════════╝

📋 Current Task:
   {self.task_description}

📁 Your Module:
   {self.module_path}

🔍 What You've Discovered:
   • Files found: {len(self.discovered_files)}
   • Patterns found: {len(self.discovered_patterns)}
   • Knowledge base results: {len(self.knowledge_base_results)}
   • Code examples: {len(self.similar_code)}
   • Tools used: {', '.join(self.tools_used) if self.tools_used else 'None yet'}

📝 Your Plan:
   • To create: {self.files_to_create if self.files_to_create else 'Not planned yet'}
   • To modify: {self.files_to_modify if self.files_to_modify else 'None'}
   • Dependencies: {self.dependencies_needed if self.dependencies_needed else 'None'}
   • Approach: {self.implementation_approach if self.implementation_approach else 'Not defined yet'}

✅ Progress:
   • Created: {self.created_files if self.created_files else 'None yet'}
   • Modified: {self.modified_files if self.modified_files else 'None yet'}

⚠️  Validation Status:
   • Path validation: {'✓ Passed' if self.path_validation_passed else '⧗ Pending'}
   • Language validation: {'✓ Passed' if self.language_validation_passed else '⧗ Pending'}
   • Placeholder check: {'✓ Passed' if self.placeholder_check_passed else '⧗ Pending'}
   • Errors to fix: {len(self.validation_errors)}

╔════════════════════════════════════════════════════════════════╗
║                     ⚠️  CRITICAL REMINDERS                     ║
╚════════════════════════════════════════════════════════════════╝

1. ALL paths MUST start with: {self.module_path}/
2. Check scratchpad.created_files before creating to avoid duplicates
3. Update scratchpad after each discovery or decision
4. NO placeholder code (TODO, etc.) - implement fully or mark blocked
"""

    def add_discovery(self, discovery_type: str, value: str):
        """Add a discovery to the scratchpad."""
        if discovery_type == "file":
            if value not in self.discovered_files:
                self.discovered_files.append(value)
        elif discovery_type == "pattern":
            self.discovered_patterns.append({"pattern": value})
        elif discovery_type == "knowledge":
            if value not in self.knowledge_base_results:
                self.knowledge_base_results.append(value)
        elif discovery_type == "code_example":
            if value not in self.similar_code:
                self.similar_code.append(value)

    def add_tool_used(self, tool_name: str):
        """Track which tools have been used."""
        if tool_name not in self.tools_used:
            self.tools_used.append(tool_name)


# =========================================================================
# Validation Gates - Error Prevention System
# =========================================================================

class ValidationGate:
    """Validation checkpoints to prevent common errors.

    Each validation gate checks for specific error patterns:
    - Path violations
    - Duplicate file creation
    - Placeholder/incomplete code
    - Language mismatches
    """

    @staticmethod
    def validate_paths(changes: List[FileChange], module_path: str) -> Tuple[bool, List[str]]:
        """Validate all paths are within module.

        Args:
            changes: List of file changes to validate
            module_path: The module's absolute path

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        module_path_str = str(module_path)

        for change in changes:
            if not change.path.startswith(module_path_str):
                errors.append(
                    f"Path violation: '{change.path}' does not start with '{module_path_str}'"
                )

        return len(errors) == 0, errors

    @staticmethod
    def validate_no_duplicates(
        changes: List[FileChange],
        scratchpad: AgentScratchpad
    ) -> Tuple[bool, List[str]]:
        """Prevent duplicate file creation.

        Args:
            changes: List of file changes to validate
            scratchpad: Agent's working memory

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        for change in changes:
            if change.action == "create" and change.path in scratchpad.created_files:
                errors.append(
                    f"Duplicate creation: '{change.path}' already in scratchpad.created_files"
                )

        return len(errors) == 0, errors

    @staticmethod
    def validate_no_placeholders(changes: List[FileChange]) -> Tuple[bool, List[str]]:
        """Check for incomplete implementations.

        Args:
            changes: List of file changes to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        placeholder_patterns = [
            "// todo",
            "# todo",
            "todo:",
            "# implementation here",
            "// implementation here",
            "# placeholder",
            "// placeholder",
            "not implemented",
            "pass  # implement",
            "raise notimplementederror",
            "throw new error(\"not implemented",
            "???",
        ]

        for change in changes:
            if change.action in ["create", "modify"] and change.content:
                content_lower = change.content.lower()
                for pattern in placeholder_patterns:
                    if pattern in content_lower:
                        errors.append(
                            f"Placeholder found in '{change.path}': contains '{pattern}'"
                        )
                        break  # One error per file is enough

        return len(errors) == 0, errors

    @staticmethod
    def validate_language(
        changes: List[FileChange],
        expected_language: str
    ) -> Tuple[bool, List[str]]:
        """Validate files use correct language.

        Args:
            changes: List of file changes to validate
            expected_language: Expected programming language

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        language_extensions = {
            "scala": [".scala"],
            "python": [".py"],
            "java": [".java"],
            "javascript": [".js", ".ts"],
            "go": [".go"],
            "rust": [".rs"],
        }

        expected_exts = language_extensions.get(expected_language.lower(), [])
        if not expected_exts:
            return True, []  # Unknown language, skip validation

        for change in changes:
            file_path = change.path

            # Skip non-code files (config, docs, etc.)
            if any(file_path.endswith(ext) for ext in [".json", ".yaml", ".yml", ".md", ".txt", ".xml", ".conf", ".sbt", ".properties"]):
                continue

            # Check if file has wrong extension
            if not any(file_path.endswith(ext) for ext in expected_exts):
                errors.append(
                    f"Language violation: '{file_path}' - expected {expected_language} file {expected_exts}"
                )

        return len(errors) == 0, errors


class ModuleAgent(BaseAgent):
    """Agent specialized for a single module.
    
    Each module agent understands:
    - The module's purpose and boundaries
    - Its public API and internal structure
    - Its dependencies and dependents
    - Its code patterns and conventions
    """
    
    def __init__(
        self,
        profile: ModuleProfile,
        llm: LLMProvider,
        repo_root: Path,
        tool_registry=None,  # Optional tool registry for tool calling
    ):
        self.profile = profile
        self.repo_root = repo_root
        self.tool_registry = tool_registry

        system_prompt = build_module_agent_prompt(profile)

        super().__init__(
            name=f"module_{profile.name}",
            llm=llm,
            system_prompt=system_prompt,
        )

    def _clean_json(self, json_str: str) -> str:
        """Clean up common JSON issues from LLM responses.

        Fixes common errors that open-source models make:
        - Unquoted property names: {status: "value"} → {"status": "value"}
        - Single quotes: {'status': 'value'} → {"status": "value"}
        - Python booleans: True/False → true/false
        - Python None: None → null
        - Trailing commas
        """
        import re

        # Fix Python-style booleans and None (must be done before quote fixing)
        json_str = re.sub(r'\bTrue\b', 'true', json_str)
        json_str = re.sub(r'\bFalse\b', 'false', json_str)
        json_str = re.sub(r'\bNone\b', 'null', json_str)

        # Replace single quotes with double quotes
        # This is tricky because we need to handle escaped quotes
        # Simple approach: replace all single quotes that aren't escaped
        json_str = json_str.replace("\\'", "___ESCAPED_QUOTE___")
        json_str = json_str.replace("'", '"')
        json_str = json_str.replace("___ESCAPED_QUOTE___", "\\'")

        # Fix unquoted property names (more complex)
        # Pattern: word characters followed by colon (not inside quotes)
        # This regex adds quotes around unquoted keys
        # Matches: word: → "word":
        json_str = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', json_str)

        # Remove trailing commas before } or ]
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        # Remove control characters
        json_str = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', json_str)

        return json_str

    def _infer_status_from_text(self, text: str) -> TaskStatus:
        """Infer task status from response text when JSON parsing fails."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['error', 'failed', 'cannot', 'unable']):
            return TaskStatus.FAILED
        elif any(word in text_lower for word in ['blocked', 'waiting', 'depends']):
            return TaskStatus.BLOCKED
        else:
            return TaskStatus.COMPLETED
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process an incoming message."""
        if message.message_type == MessageType.TASK_ASSIGNMENT:
            result = await self.execute_task(
                SubTask(**message.payload)
            )
            return AgentMessage(
                id=str(uuid.uuid4()),
                sender=self.name,
                recipient=message.sender,
                message_type=MessageType.TASK_RESULT,
                payload=result.model_dump(),
                correlation_id=message.correlation_id,
            )
        
        elif message.message_type == MessageType.QUERY:
            response = await self.think(message.payload.get("question", ""))
            return AgentMessage(
                id=str(uuid.uuid4()),
                sender=self.name,
                recipient=message.sender,
                message_type=MessageType.RESPONSE,
                payload={"answer": response},
                correlation_id=message.correlation_id,
            )
        
        # Default: just acknowledge
        return AgentMessage(
            id=str(uuid.uuid4()),
            sender=self.name,
            recipient=message.sender,
            message_type=MessageType.STATUS_UPDATE,
            payload={"status": "acknowledged"},
            correlation_id=message.correlation_id,
        )
    
    async def execute_task(self, task: SubTask) -> SubTaskResult:
        """Execute a subtask within this module's boundaries."""
        # Use tool calling with scratchpad if available
        if self.tool_registry:
            return await self._execute_with_tools_and_scratchpad(task)
        else:
            return await self._execute_without_tools(task)

    async def _execute_without_tools(self, task: SubTask, json_retry_count: int = 0) -> SubTaskResult:
        """Execute task without tool calling (original behavior)."""
        is_empty = self.profile.file_count == 0

        # Add JSON retry warning if this is a JSON retry
        json_retry_warning = ""
        if json_retry_count > 0:
            json_retry_warning = f"""

╔════════════════════════════════════════════════════════════════════════════╗
║          🚨 JSON FORMATTING ERROR - RETRY #{json_retry_count} 🚨                    ║
╚════════════════════════════════════════════════════════════════════════════╝

Your previous response had INVALID JSON formatting. Common errors:
  • Property names MUST be in double quotes: {{"status": "completed"}} ✓
  • Single quotes are INVALID: {{'status': 'completed'}} ✗
  • Unquoted property names are INVALID: {{status: "completed"}} ✗
  • Missing commas between properties
  • Trailing commas before closing braces

CRITICAL REQUIREMENTS FOR THIS RESPONSE:
1. Return ONLY valid JSON - no extra text before or after
2. ALL property names MUST be in DOUBLE QUOTES
3. ALL string values MUST be in DOUBLE QUOTES
4. Use proper JSON syntax with commas between items
5. Test your JSON mentally before responding

⚠️  If you return invalid JSON again, the task will FAIL! ⚠️

"""

        # Build the task prompt
        prompt = f"""{json_retry_warning}
Execute this task within the {self.profile.name} module:

Task: {task.description}

Module Status: {"This module is EMPTY (0 files). You will be CREATING initial files." if is_empty else f"Existing module with {self.profile.file_count} files"}
Affected files: {', '.join(task.affected_files) if task.affected_files else 'Determine from task'}

Requirements:
1. Only create/modify files within this module ({self.profile.path})
2. {"CREATE new files as needed - empty modules need bootstrapping" if is_empty else "Preserve existing public API unless explicitly changing it"}
3. {"Establish good patterns for this new module" if is_empty else "Follow existing code patterns"}
4. Include any necessary tests

Respond with a JSON object:
```json
{{
  "status": "completed" | "failed" | "blocked",
  "changes": [
    {{
      "path": "relative/path/to/file",
      "action": "create" | "modify" | "delete",
      "content": "full file content for create/modify",
      "diff": "optional diff description"
    }}
  ],
  "tests_added": ["list of test file paths"],
  "blockers": ["list of issues requiring other modules, if any"],
  "notes": "implementation notes"
}}
```

{"IMPORTANT: Empty modules are NOT blockers. Create the initial files as specified in the task." if is_empty else ""}
"""

        response = await self.think(prompt)

        # Use shared parsing logic
        result = self._parse_task_result(task, response, is_empty)

        # Check if JSON parsing failed and retry if allowed
        if result.error and "JSON decode error" in result.error:
            max_json_retries = 2  # Allow up to 2 JSON retries

            if json_retry_count < max_json_retries:
                # Log the retry attempt
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"JSON parsing failed (attempt {json_retry_count + 1}/{max_json_retries + 1}), "
                    f"retrying with enhanced JSON instructions..."
                )

                # Retry with enhanced JSON instructions
                return await self._execute_without_tools(
                    task=task,
                    json_retry_count=json_retry_count + 1
                )
            else:
                # Max JSON retries exceeded - add helpful note
                result.notes = (
                    f"Failed after {max_json_retries + 1} attempts to generate valid JSON.\n\n"
                    f"Original error: {result.error}\n\n"
                    f"This usually indicates the LLM model has difficulty with JSON formatting. "
                    f"Consider using a different model or provider.\n\n"
                    f"{result.notes}"
                )

        return result

    async def _execute_with_tools_and_scratchpad(self, task: SubTask, max_retries: int = 2) -> SubTaskResult:
        """Execute task with tool calling and scratchpad support with retry logic.

        This method:
        1. Initializes a scratchpad for working memory
        2. Uses validation gates to check for common errors
        3. Automatically retries on fixable validation errors
        4. Provides feedback to agent based on scratchpad state

        Args:
            task: The subtask to execute
            max_retries: Maximum retry attempts (default: 2)

        Returns:
            SubTaskResult with success/failure status
        """
        # Initialize scratchpad
        scratchpad = AgentScratchpad(
            task_id=task.id,
            task_description=task.description,
            module_path=str(self.profile.path),
        )

        for attempt in range(max_retries + 1):
            # Execute the task with tools
            result = await self._execute_with_tools(task, scratchpad=scratchpad, attempt=attempt)

            # If task failed or blocked without changes, return immediately
            if result.status in [TaskStatus.FAILED, TaskStatus.BLOCKED] and not result.changes:
                return result

            # Run validation gates
            all_errors = []

            # Gate 1: Path validation
            path_ok, path_errors = ValidationGate.validate_paths(result.changes, str(self.profile.path))
            if path_ok:
                scratchpad.path_validation_passed = True
            else:
                all_errors.extend(path_errors)

            # Gate 2: No duplicates
            dup_ok, dup_errors = ValidationGate.validate_no_duplicates(result.changes, scratchpad)
            if not dup_ok:
                all_errors.extend(dup_errors)

            # Gate 3: No placeholders (only if completed)
            if result.status == TaskStatus.COMPLETED:
                placeholder_ok, placeholder_errors = ValidationGate.validate_no_placeholders(result.changes)
                if placeholder_ok:
                    scratchpad.placeholder_check_passed = True
                else:
                    all_errors.extend(placeholder_errors)

            # Gate 4: Language validation
            lang_ok, lang_errors = ValidationGate.validate_language(result.changes, self.profile.language)
            if lang_ok:
                scratchpad.language_validation_passed = True
            else:
                all_errors.extend(lang_errors)

            # If all validations passed, success!
            if not all_errors:
                # Update scratchpad with created files for future reference
                for change in result.changes:
                    if change.action == "create":
                        scratchpad.created_files.append(change.path)
                    elif change.action == "modify":
                        scratchpad.modified_files.append(change.path)

                # Add scratchpad summary to notes
                result.notes = f"Scratchpad Summary:\n" \
                             f"- Tools used: {', '.join(scratchpad.tools_used)}\n" \
                             f"- Discoveries: {len(scratchpad.knowledge_base_results)} KB results, " \
                             f"{len(scratchpad.discovered_files)} files, " \
                             f"{len(scratchpad.discovered_patterns)} patterns\n" \
                             f"- Validations: All passed ✓\n\n" \
                             f"{result.notes}"

                return result

            # Validations failed - check if we should retry
            if attempt < max_retries:
                # Add errors to scratchpad for next attempt
                scratchpad.validation_errors.extend(all_errors)

                # Add retry info to result notes for transparency
                result.notes = f"[Attempt {attempt + 1}/{max_retries + 1}] Validation failed, retrying...\n" \
                             f"Errors: {'; '.join(all_errors[:3])}\n\n{result.notes}"

                # Continue to next attempt with updated scratchpad
                continue
            else:
                # Max retries exceeded - return failure with details
                result.status = TaskStatus.FAILED
                result.error = (
                    f"Validation failed after {max_retries + 1} attempts.\n\n"
                    f"Validation Errors:\n"
                    + "\n".join(f"  • {e}" for e in all_errors[:5])
                )
                result.notes = (
                    f"Scratchpad Summary:\n"
                    f"- Tools used: {', '.join(scratchpad.tools_used)}\n"
                    f"- Path validation: {'✓' if scratchpad.path_validation_passed else '✗'}\n"
                    f"- Language validation: {'✓' if scratchpad.language_validation_passed else '✗'}\n"
                    f"- Placeholder check: {'✓' if scratchpad.placeholder_check_passed else '✗'}\n"
                    f"- Total errors: {len(all_errors)}\n\n"
                    f"{result.notes}"
                )

                return result

        # Should not reach here, but return failure if it does
        return SubTaskResult(
            subtask_id=task.id,
            status=TaskStatus.FAILED,
            error="Unexpected: exceeded retry loop without returning",
        )

    async def analyze_impact(self, description: str) -> dict:
        """Analyze how a change might impact this module."""
        prompt = f"""Analyze how this change might affect the {self.profile.name} module:

Change description: {description}

Consider:
1. Which files in this module would need changes?
2. Would this require API changes?
3. What tests would need updating?
4. Are there any risks or concerns?

Respond with JSON:
```json
{{
  "affected_files": ["list of files"],
  "api_changes": true/false,
  "test_updates_needed": ["list of test files"],
  "risks": ["list of concerns"],
  "effort_estimate": "low" | "medium" | "high"
}}
```
"""
        
        response = await self.think(prompt)
        
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        
        return {
            "affected_files": [],
            "api_changes": False,
            "test_updates_needed": [],
            "risks": ["Could not analyze impact"],
            "effort_estimate": "unknown",
        }

    async def _execute_with_tools(
        self,
        task: SubTask,
        scratchpad: AgentScratchpad | None = None,
        attempt: int = 0,
        json_retry_count: int = 0
    ) -> SubTaskResult:
        """Execute task with tool calling support.

        Args:
            task: The subtask to execute
            scratchpad: Optional scratchpad for tracking state (recommended)
            attempt: Current attempt number (for retry logic)
            json_retry_count: Number of JSON parsing retries attempted

        Returns:
            SubTaskResult with changes and status
        """
        is_empty = self.profile.file_count == 0

        # Build list of available tools
        available_tools = []
        if self.tool_registry:
            for tool in self.tool_registry.list_tools():
                available_tools.append(f"- {tool.name}: {tool.description}")

        tools_section = "\n".join(available_tools) if available_tools else "- read_file, grep_codebase (basic tools)"

        # Add scratchpad context if available
        scratchpad_context = ""
        if scratchpad:
            scratchpad_context = scratchpad.to_prompt_context()

            # Add retry feedback if this is not the first attempt
            if attempt > 0:
                scratchpad_context += f"""

╔════════════════════════════════════════════════════════════════╗
║                  ⚠️  RETRY ATTEMPT #{attempt + 1}                    ║
╚════════════════════════════════════════════════════════════════╝

Previous attempt had validation errors:
{chr(10).join(f"  • {e}" for e in scratchpad.validation_errors[-5:])}

FIX THESE ERRORS in this attempt by:
1. Checking your scratchpad to see what went wrong
2. Ensuring ALL paths start with {scratchpad.module_path}/
3. Not creating files already in scratchpad.created_files
4. Implementing complete code (no TODOs)
"""

        # Add JSON retry warning if this is a JSON retry
        json_retry_warning = ""
        if json_retry_count > 0:
            json_retry_warning = f"""

╔════════════════════════════════════════════════════════════════════════════╗
║          🚨 JSON FORMATTING ERROR - RETRY #{json_retry_count} 🚨                    ║
╚════════════════════════════════════════════════════════════════════════════╝

Your previous response had INVALID JSON formatting. Common errors:
  • Property names MUST be in double quotes: {{"status": "completed"}} ✓
  • Single quotes are INVALID: {{'status': 'completed'}} ✗
  • Unquoted property names are INVALID: {{status: "completed"}} ✗
  • Missing commas between properties
  • Trailing commas before closing braces

CRITICAL REQUIREMENTS FOR THIS RESPONSE:
1. Return ONLY valid JSON - no extra text before or after
2. ALL property names MUST be in DOUBLE QUOTES
3. ALL string values MUST be in DOUBLE QUOTES
4. Use proper JSON syntax with commas between items
5. Test your JSON mentally before responding

Example of CORRECT JSON format:
```json
{{{{
  "status": "completed",
  "changes": [
    {{{{
      "path": "/full/absolute/path/to/file.scala",
      "action": "create",
      "content": "// actual file content here"
    }}}}
  ],
  "tests_added": [],
  "blockers": [],
  "notes": "Implementation complete"
}}}}
```

⚠️  If you return invalid JSON again, the task will FAIL! ⚠️
"""

        # Build enhanced prompt with tool instructions
        prompt = f"""{json_retry_warning}{scratchpad_context}

Execute this task within the {self.profile.name} module:

Task: {task.description}

Module Status: {"This module is EMPTY (0 files). You will be CREATING initial files." if is_empty else f"Existing module with {self.profile.file_count} files"}
Affected files: {', '.join(task.affected_files) if task.affected_files else 'Determine from task'}

You have access to these tools to gather context:
{tools_section}

🔥 CRITICAL: NEVER BLOCK ON MISSING DETAILS - SEARCH INSTEAD! 🔥

If task description is vague or missing details (e.g., "create repository" without method signatures):
1. 🔍 Use search_knowledge_base to find similar repositories/services/models in the codebase
2. 🔍 Use grep_codebase to search for existing patterns (e.g., grep for "Repository" or "Service")
3. 🔍 Use read_file to examine similar files and understand the pattern
4. 📝 INFER the interface from similar code - copy the pattern
5. ✅ Implement based on discovered patterns - DON'T BLOCK!

Example workflow:
- Task says: "Create UserRepository"
- DON'T block saying "I need method details"
- DO: search_knowledge_base(query="repository CRUD methods")
- DO: grep_codebase(pattern="trait.*Repository", file_pattern="*.scala")
- DO: read_file on similar repository to see the pattern
- DO: Implement UserRepository following that pattern

MANDATORY TOOL USAGE BEFORE IMPLEMENTING:
1. Use read_file to examine existing code in affected files
2. Use search_knowledge_base to find similar implementations in the codebase
3. Use grep_codebase to search for existing patterns (repositories, services, models)
4. Implement changes following discovered patterns
5. Use validate_syntax (if available) to check your code
6. Use run_tests (if available) to validate

═══════════════════════════════════════════════════════════════════════
🚨 CRITICAL: FILE PATH REQUIREMENT - READ THIS FIRST 🚨
═══════════════════════════════════════════════════════════════════════

YOUR MODULE'S ABSOLUTE PATH IS: {self.profile.path}

ALL file paths in your response MUST start with this EXACT path:
  {self.profile.path}/

✅ CORRECT examples:
  - {self.profile.path}/src/main/scala/MyClass.scala
  - {self.profile.path}/src/test/scala/MyClassSpec.scala
  - {self.profile.path}/README.md

❌ WRONG examples (will cause PATH VIOLATION error):
  - src/main/scala/MyClass.scala          (missing module path!)
  - MyClass.scala                         (missing full path!)
  - {self.profile.name}/src/MyClass.scala (wrong - use absolute path!)

The system will REJECT any files that don't start with: {self.profile.path}/

═══════════════════════════════════════════════════════════════════════

Requirements:
1. Only create/modify files within this module ({self.profile.path})
2. {"CREATE new files as needed - empty modules need bootstrapping" if is_empty else "Preserve existing public API unless explicitly changing it"}
3. {"Establish good patterns for this new module" if is_empty else "Follow existing code patterns"}
4. Include any necessary tests

After using tools to gather context, respond with a JSON object:
```json
{{
  "status": "completed" | "failed" | "blocked",
  "changes": [
    {{
      "path": "{self.profile.path}/path/to/YourFile.ext",  // ⚠️ MUST start with {self.profile.path}/
      "action": "create" | "modify" | "delete",
      "content": "full file content for create/modify",
      "diff": "optional diff description"
    }}
  ],
  "tests_added": ["{self.profile.path}/path/to/tests"],  // ⚠️ MUST start with {self.profile.path}/
  "blockers": ["list of issues requiring other modules, if any"],
  "notes": "implementation notes",
  "tools_used": ["list of tools you called"]
}}
```

🔴 REMINDER: ALL PATHS MUST BE ABSOLUTE 🔴
- Your module path: {self.profile.path}
- Every file path must start with: {self.profile.path}/
- Use read_file tool FIRST to see existing file paths and match that exact structure
- Example: "{self.profile.path}/src/main/scala/Example.scala" ✓
- Example: "src/main/scala/Example.scala" ✗ REJECTED!

CRITICAL IMPLEMENTATION REQUIREMENTS:
- NO placeholder code like "// TODO: implement", "# Implementation here", "pass"
- Write COMPLETE, WORKING implementations
- If you can't implement something fully, mark status as "blocked" with reason
- NEVER mark as "completed" if code contains TODOs or placeholders

{"IMPORTANT: Empty modules are NOT blockers. Create the initial files as specified in the task." if is_empty else ""}
"""

        # Get tools in appropriate format
        from modular_agents.llm.base import LLMMessage

        tools = self.tool_registry.to_anthropic_format()  # Use Anthropic format for Claude
        if self.llm.name == "openai":
            tools = self.tool_registry.to_openai_format()

        # Convert conversation history to dict format
        messages = []
        for msg in self.conversation_history:
            messages.append({"role": msg.role, "content": msg.content})

        # Add task prompt as user message
        messages.append({"role": "user", "content": prompt})

        # Tool calling loop
        max_rounds = 10  # Prevent infinite loops
        all_tool_results = []

        for round_num in range(max_rounds):
            # Call LLM with tools
            response_text, tool_calls = await self.llm.chat_with_tools(
                messages=messages,
                system=self.system_prompt,
                tools=tools,
                max_tool_rounds=1,  # One tool call at a time
            )

            # If no tool calls, we have final response
            if not tool_calls:
                # Parse final response
                result = self._parse_task_result(task, response_text, is_empty)

                # Check if JSON parsing failed and retry if allowed
                if result.error and "JSON decode error" in result.error:
                    max_json_retries = 2  # Allow up to 2 JSON retries

                    if json_retry_count < max_json_retries:
                        # Log the retry attempt (will be visible in debug mode)
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(
                            f"JSON parsing failed (attempt {json_retry_count + 1}/{max_json_retries + 1}), "
                            f"retrying with enhanced JSON instructions..."
                        )

                        # Retry with enhanced JSON instructions
                        return await self._execute_with_tools(
                            task=task,
                            scratchpad=scratchpad,
                            attempt=attempt,
                            json_retry_count=json_retry_count + 1
                        )
                    else:
                        # Max JSON retries exceeded - add helpful note
                        result.notes = (
                            f"Failed after {max_json_retries + 1} attempts to generate valid JSON.\n\n"
                            f"Original error: {result.error}\n\n"
                            f"This usually indicates the LLM model has difficulty with JSON formatting. "
                            f"Consider using a different model or provider.\n\n"
                            f"{result.notes}"
                        )

                return result

            # Execute tool calls
            for tool_call in tool_calls:
                # Track tool usage in scratchpad
                if scratchpad:
                    scratchpad.add_tool_used(tool_call.name)

                # Execute tool
                result = await self.tool_registry.execute(tool_call)
                all_tool_results.append((tool_call, result))

                # Update scratchpad with discoveries from specific tools
                if scratchpad and result.success:
                    if tool_call.name == "search_knowledge_base":
                        scratchpad.add_discovery("knowledge", result.output[:200])
                    elif tool_call.name == "read_file":
                        file_path = tool_call.parameters.get("file_path", "unknown")
                        scratchpad.add_discovery("file", file_path)
                    elif tool_call.name == "grep_codebase":
                        scratchpad.add_discovery("pattern", f"grep: {tool_call.parameters.get('pattern', 'unknown')}")

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

        # Max rounds exceeded
        return SubTaskResult(
            subtask_id=task.id,
            status=TaskStatus.FAILED,
            error=f"Tool calling exceeded maximum rounds ({max_rounds})",
            notes=f"Used {len(all_tool_results)} tools but did not reach final response",
        )

    def _parse_task_result(self, task: SubTask, response: str, is_empty: bool) -> SubTaskResult:
        """Parse task result from LLM response (shared by both tool and non-tool execution)."""
        try:
            # Extract JSON
            content = response
            start = content.find('{')
            end = content.rfind('}') + 1

            if start >= 0 and end > start:
                json_str = content[start:end]
                # Try to clean JSON before parsing
                json_str = self._clean_json(json_str)
                result_data = json.loads(json_str)

                result = SubTaskResult(
                    subtask_id=task.id,
                    status=TaskStatus(result_data.get("status", "completed")),
                    changes=[
                        FileChange(**c) for c in result_data.get("changes", [])
                    ],
                    tests_added=result_data.get("tests_added", []),
                    blockers=result_data.get("blockers", []),
                    notes=result_data.get("notes", ""),
                )

                # Post-process: detect if agent incorrectly blocked on empty module
                if (result.status == TaskStatus.BLOCKED and
                    is_empty and
                    result.blockers and
                    any("no files" in b.lower() or "empty" in b.lower() or "no code" in b.lower()
                        for b in result.blockers)):
                    # Convert to failed with helpful message
                    result.status = TaskStatus.FAILED
                    result.error = (
                        "Agent incorrectly reported blocker for empty module. "
                        "Empty modules should be bootstrapped with initial files, not blocked. "
                        "This may indicate the agent needs clearer instructions or the task needs more specificity."
                    )
                    result.notes = f"Original blockers: {', '.join(result.blockers)}\n\n{result.notes}"

                # CRITICAL: Validate language consistency
                expected_lang = self.profile.language
                language_extensions = {
                    "scala": [".scala"],
                    "python": [".py"],
                    "java": [".java"],
                    "javascript": [".js", ".ts"],
                    "go": [".go"],
                    "rust": [".rs"],
                }

                expected_exts = language_extensions.get(expected_lang.lower(), [])
                wrong_files = []
                wrong_paths = []
                placeholder_files = []

                module_path_str = str(self.profile.path)

                for change in result.changes:
                    file_path = change.path

                    # Validate path is within module
                    if not file_path.startswith(module_path_str):
                        wrong_paths.append(file_path)

                    # Validate language extension
                    if expected_exts:
                        # Check if file has wrong extension
                        if not any(file_path.endswith(ext) for ext in expected_exts):
                            # Skip non-code files (config, docs, etc.)
                            if any(file_path.endswith(ext) for ext in [".json", ".yaml", ".yml", ".md", ".txt", ".xml", ".conf"]):
                                continue
                            wrong_files.append(file_path)

                    # Validate no placeholder code (only for create/modify actions)
                    if change.action in ["create", "modify"] and change.content:
                        content_lower = change.content.lower()
                        placeholder_indicators = [
                            "// todo",
                            "# todo",
                            "todo:",
                            "# implementation here",
                            "// implementation here",
                            "# placeholder",
                            "// placeholder",
                            "not implemented",
                            "pass  # implement",
                            "raise notimplementederror",
                            "throw new error(\"not implemented",
                        ]
                        if any(indicator in content_lower for indicator in placeholder_indicators):
                            placeholder_files.append(file_path)

                # Check for path violations
                if wrong_paths:
                    result.status = TaskStatus.FAILED
                    result.error = (
                        f"PATH VIOLATION: Files created outside module directory!\n"
                        f"Module path: {module_path_str}\n"
                        f"Wrong paths: {', '.join(wrong_paths)}\n\n"
                        f"ALL file paths must start with the module path.\n"
                        f"Use read_file tool to see existing file structure and match it."
                    )
                    result.notes = f"Path violation detected.\n\nOriginal notes: {result.notes}"
                    return result

                # Check for language violations
                if wrong_files:
                    result.status = TaskStatus.FAILED
                    result.error = (
                        f"LANGUAGE VIOLATION: Agent created files in wrong language!\n"
                        f"Expected: {expected_lang} files {expected_exts}\n"
                        f"Got wrong files: {', '.join(wrong_files)}\n\n"
                        f"The module '{self.profile.name}' is a {expected_lang} module. "
                        f"All code files MUST use {expected_lang}. "
                        f"This is a critical error that violates repository consistency."
                    )
                    result.notes = f"Language violation detected.\n\nOriginal notes: {result.notes}"
                    return result

                # Check for placeholder code
                if placeholder_files and result.status == TaskStatus.COMPLETED:
                    result.status = TaskStatus.FAILED
                    result.error = (
                        f"INCOMPLETE IMPLEMENTATION: Code contains placeholders/TODOs!\n"
                        f"Files with placeholders: {', '.join(placeholder_files)}\n\n"
                        f"Do NOT mark as 'completed' if implementation contains TODOs or placeholders.\n"
                        f"Either:\n"
                        f"1. Complete the full implementation, OR\n"
                        f"2. Mark as 'blocked' and explain what you need\n\n"
                        f"Use tools (read_file, search_knowledge_base) to find examples and implement fully."
                    )
                    result.notes = f"Placeholder code detected.\n\nOriginal notes: {result.notes}"
                    return result

                return result
            else:
                # No JSON found in response
                return SubTaskResult(
                    subtask_id=task.id,
                    status=TaskStatus.FAILED,
                    error="No JSON object found in response",
                    notes=f"Raw response:\n{response[:1000]}",
                )
        except json.JSONDecodeError as e:
            # JSON parsing failed - show the problematic JSON
            error_msg = f"JSON decode error at line {e.lineno} col {e.colno}: {e.msg}"
            return SubTaskResult(
                subtask_id=task.id,
                status=self._infer_status_from_text(response),
                error=error_msg,
                notes=f"Attempted to parse:\n{json_str[:1000] if 'json_str' in locals() else response[:1000]}",
            )
        except (KeyError, ValueError) as e:
            return SubTaskResult(
                subtask_id=task.id,
                status=self._infer_status_from_text(response),
                error=f"Failed to parse response: {e}",
                notes=f"Raw response:\n{response[:1000]}",
            )
        except Exception as e:
            # Catch-all for unexpected errors
            return SubTaskResult(
                subtask_id=task.id,
                status=TaskStatus.FAILED,
                error=f"Unexpected error: {type(e).__name__}: {e}",
                notes=f"Raw response:\n{response[:1000]}",
            )

    # =========================================================================
    # State Management for Checkpointing
    # =========================================================================

    def save_state(self) -> "AgentState":  # type: ignore
        """Save current agent state for checkpointing.

        Returns:
            AgentState containing conversation history and metadata
        """
        from modular_agents.checkpoint import AgentState, LLMMessage

        # Convert conversation history to checkpoint format
        checkpoint_history = []
        for msg in self.conversation_history:
            checkpoint_msg = LLMMessage(
                role=msg.role,
                content=msg.content,
            )
            checkpoint_history.append(checkpoint_msg)

        return AgentState(
            agent_name=self.name,
            conversation_history=checkpoint_history,
            retry_count={},  # Retry counts tracked at orchestrator level
            last_error=None,
            metadata={
                "module_name": self.profile.name,
                "module_path": str(self.profile.path),
            },
        )

    def restore_state(self, state: "AgentState") -> None:  # type: ignore
        """Restore agent state from checkpoint.

        Args:
            state: AgentState to restore from
        """
        from modular_agents.llm.base import LLMMessage as BaseLLMMessage

        # Clear current conversation
        self.conversation_history = []

        # Restore conversation history
        for msg in state.conversation_history:
            base_msg = BaseLLMMessage(
                role=msg.role,
                content=msg.content,
            )
            self.conversation_history.append(base_msg)
