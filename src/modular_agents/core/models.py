"""Core data models for the multi-agent framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================


class ProjectType(str, Enum):
    """Supported project types."""
    SBT = "sbt"
    MAVEN = "maven"
    GRADLE = "gradle"
    NPM = "npm"
    CARGO = "cargo"
    POETRY = "poetry"
    GO = "go"
    UNKNOWN = "unknown"


class TaskStatus(str, Enum):
    """Status of a task or subtask."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class MessageType(str, Enum):
    """Types of inter-agent messages."""
    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    QUERY = "query"
    RESPONSE = "response"
    STATUS_UPDATE = "status_update"
    ERROR = "error"


# =============================================================================
# Module & Repository Models
# =============================================================================


class ModuleProfile(BaseModel):
    """Profile of a single module in the codebase."""

    name: str
    path: Path
    purpose: str = ""
    language: str = "unknown"  # Primary language (scala, python, java, etc.)
    framework: str = ""  # Framework if detected (akka, spring, flask, etc.)
    packages: list[str] = Field(default_factory=list)
    public_api: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)  # Internal module deps
    external_deps: list[str] = Field(default_factory=list)  # External library deps
    dependents: list[str] = Field(default_factory=list)  # Modules that depend on this
    test_patterns: list[str] = Field(default_factory=list)
    file_count: int = 0
    loc: int = 0  # Lines of code
    code_examples: list[str] = Field(default_factory=list)  # Example code snippets
    naming_patterns: list[str] = Field(default_factory=list)  # Naming conventions

    class Config:
        arbitrary_types_allowed = True


class RepoKnowledge(BaseModel):
    """Complete knowledge about a repository."""
    
    root_path: Path
    project_type: ProjectType
    modules: list[ModuleProfile] = Field(default_factory=list)
    dependency_graph: dict[str, list[str]] = Field(default_factory=dict)
    build_file: str = ""
    analyzed_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True
    
    def get_module(self, name: str) -> ModuleProfile | None:
        """Get a module by name."""
        for m in self.modules:
            if m.name == name:
                return m
        return None
    
    def get_affected_modules(self, file_paths: list[str]) -> list[str]:
        """Determine which modules are affected by changes to given files."""
        affected = set()
        for file_path in file_paths:
            for module in self.modules:
                if file_path.startswith(str(module.path)):
                    affected.add(module.name)
        return list(affected)


# =============================================================================
# Task Models
# =============================================================================


class SubTask(BaseModel):
    """A subtask assigned to a specific module agent."""
    
    id: str
    module: str
    description: str
    affected_files: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)  # Other subtask IDs
    priority: int = 1
    status: TaskStatus = TaskStatus.PENDING
    
    
class Task(BaseModel):
    """A high-level task from the user."""
    
    id: str
    description: str
    subtasks: list[SubTask] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None


class FileChange(BaseModel):
    """A change to a file."""
    
    path: str
    action: str  # "create", "modify", "delete"
    content: str | None = None
    diff: str | None = None


class SubTaskResult(BaseModel):
    """Result of executing a subtask."""
    
    subtask_id: str
    status: TaskStatus
    changes: list[FileChange] = Field(default_factory=list)
    tests_added: list[str] = Field(default_factory=list)
    tests_passed: bool = True
    error: str | None = None
    blockers: list[str] = Field(default_factory=list)
    notes: str = ""


class TaskResult(BaseModel):
    """Final result of a complete task."""
    
    task_id: str
    status: TaskStatus
    subtask_results: list[SubTaskResult] = Field(default_factory=list)
    integration_passed: bool = True
    summary: str = ""
    error: str | None = None


# =============================================================================
# Agent Communication
# =============================================================================


class AgentMessage(BaseModel):
    """Message passed between agents."""
    
    id: str
    sender: str
    recipient: str  # Agent name or "broadcast"
    message_type: MessageType
    payload: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = None  # To track related messages
    timestamp: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Execution Plan
# =============================================================================


class ExecutionPhase(BaseModel):
    """A phase of execution containing parallelizable subtasks."""
    
    phase_number: int
    subtask_ids: list[str]


class ExecutionPlan(BaseModel):
    """Plan for executing a task's subtasks."""

    task_id: str
    phases: list[ExecutionPhase] = Field(default_factory=list)

    def get_phase_for_subtask(self, subtask_id: str) -> int | None:
        """Get the phase number for a subtask."""
        for phase in self.phases:
            if subtask_id in phase.subtask_ids:
                return phase.phase_number
        return None


# =============================================================================
# Knowledge Base Models
# =============================================================================


class CodeChunk(BaseModel):
    """A chunk of code with metadata and embeddings."""

    id: str  # UUID
    repo_path: str  # Repository root path
    file_path: str  # Relative file path within repo
    module_name: str  # Module this chunk belongs to
    language: str  # Programming language
    chunk_type: str  # "class", "function", "interface", "config", etc.
    name: str  # Name of the entity (class name, function name, etc.)
    content: str  # Actual source code
    start_line: int
    end_line: int
    embedding: list[float] | None = None  # Vector embedding
    summary: str = ""  # LLM-generated summary
    purpose: str = ""  # What this code does
    dependencies: list[str] = Field(default_factory=list)  # Other chunks this depends on
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class ChunkRelation(BaseModel):
    """Relationship between code chunks."""

    source_chunk_id: str
    target_chunk_id: str
    relation_type: str  # "imports", "calls", "extends", "implements", "uses"
    weight: float = 1.0  # Strength of relationship


class IndexedRepo(BaseModel):
    """Metadata about an indexed repository."""

    repo_path: str  # Absolute path to repository
    project_type: ProjectType
    language: str  # Primary language
    total_chunks: int = 0
    total_files: int = 0
    indexed_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)


class TaskLearning(BaseModel):
    """Learning captured from completed tasks."""

    id: str  # UUID
    task_description: str
    repo_path: str
    module_names: list[str] = Field(default_factory=list)  # Modules involved
    patterns_learned: list[str] = Field(default_factory=list)  # Patterns discovered
    code_chunks: list[str] = Field(default_factory=list)  # Chunk IDs created/modified
    success: bool = True
    error_message: str = ""
    completed_at: datetime = Field(default_factory=datetime.now)
