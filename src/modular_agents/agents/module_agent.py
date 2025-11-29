"""Module agent - specialist for a single module."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

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
    ):
        self.profile = profile
        self.repo_root = repo_root
        
        system_prompt = build_module_agent_prompt(profile)
        
        super().__init__(
            name=f"module_{profile.name}",
            llm=llm,
            system_prompt=system_prompt,
        )

    def _clean_json(self, json_str: str) -> str:
        """Clean up common JSON issues from LLM responses."""
        import re
        
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
        is_empty = self.profile.file_count == 0

        # Build the task prompt
        prompt = f"""Execute this task within the {self.profile.name} module:

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
        
        # Parse the response
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

                for change in result.changes:
                    file_path = change.path
                    if expected_exts:
                        # Check if file has wrong extension
                        if not any(file_path.endswith(ext) for ext in expected_exts):
                            # Skip non-code files (config, docs, etc.)
                            if any(file_path.endswith(ext) for ext in [".json", ".yaml", ".yml", ".md", ".txt", ".xml", ".conf"]):
                                continue
                            wrong_files.append(file_path)

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
