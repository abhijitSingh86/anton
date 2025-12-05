"""Tools for code reading and searching."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .base import Tool, ToolResult


class ReadFileTool(Tool):
    """Read contents of a file in the repository."""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read the contents of a file in the repository. "
            "Use this to examine code before making changes or to understand context."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file relative to repository root",
                },
                "start_line": {
                    "type": "integer",
                    "description": "Optional: Start line number (1-indexed)",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Optional: End line number (inclusive)",
                },
            },
            "required": ["file_path"],
        }

    async def execute(
        self,
        file_path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        **kwargs,
    ) -> ToolResult:
        """Read file contents."""
        try:
            full_path = (self.repo_path / file_path).resolve()

            # Security: Ensure path is within repo
            if not str(full_path).startswith(str(self.repo_path)):
                return ToolResult(
                    tool_call_id="",
                    tool_name=self.name,
                    success=False,
                    output="",
                    error=f"Path '{file_path}' is outside repository",
                )

            if not full_path.exists():
                return ToolResult(
                    tool_call_id="",
                    tool_name=self.name,
                    success=False,
                    output="",
                    error=f"File not found: {file_path}",
                )

            # Read file
            with open(full_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Apply line range if specified
            if start_line is not None or end_line is not None:
                start = (start_line - 1) if start_line else 0
                end = end_line if end_line else len(lines)
                lines = lines[start:end]
                line_numbers = range(start + 1, start + len(lines) + 1)
                content = "".join(
                    f"{num:4d} | {line}" for num, line in zip(line_numbers, lines)
                )
            else:
                content = "".join(lines)

            return ToolResult(
                tool_call_id="",
                tool_name=self.name,
                success=True,
                output=content,
                metadata={
                    "file_path": file_path,
                    "total_lines": len(lines),
                },
            )

        except UnicodeDecodeError:
            return ToolResult(
                tool_call_id="",
                tool_name=self.name,
                success=False,
                output="",
                error=f"File '{file_path}' is not a text file (binary or encoding issue)",
            )
        except Exception as e:
            return ToolResult(
                tool_call_id="",
                tool_name=self.name,
                success=False,
                output="",
                error=f"Failed to read file: {str(e)}",
            )


class GrepCodebaseTool(Tool):
    """Search for patterns in the codebase using grep."""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()

    @property
    def name(self) -> str:
        return "grep_codebase"

    @property
    def description(self) -> str:
        return (
            "Search for a pattern in the codebase using grep. "
            "Useful for finding where functions/classes are defined or used. "
            "Supports regex patterns."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Pattern to search for (regex supported)",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional: File pattern to filter (e.g., '*.py', '*.scala')",
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "Optional: Case-insensitive search (default: false)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Optional: Maximum number of results (default: 50)",
                },
            },
            "required": ["pattern"],
        }

    async def execute(
        self,
        pattern: str,
        file_pattern: str | None = None,
        ignore_case: bool = False,
        max_results: int = 50,
        **kwargs,
    ) -> ToolResult:
        """Search codebase for pattern."""
        try:
            # Build grep command
            cmd = ["grep", "-rn"]  # recursive, line numbers

            if ignore_case:
                cmd.append("-i")

            # Add pattern
            cmd.append(pattern)

            # Add file pattern if specified
            if file_pattern:
                cmd.extend(["--include", file_pattern])

            # Exclude common directories
            cmd.extend(
                [
                    "--exclude-dir=.git",
                    "--exclude-dir=node_modules",
                    "--exclude-dir=venv",
                    "--exclude-dir=__pycache__",
                    "--exclude-dir=.modular-agents",
                ]
            )

            # Add repository path
            cmd.append(str(self.repo_path))

            # Execute grep
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                # Limit results
                if len(lines) > max_results:
                    lines = lines[:max_results]
                    output = "\n".join(lines) + f"\n\n... ({len(lines) - max_results} more matches omitted)"
                else:
                    output = "\n".join(lines)

                return ToolResult(
                    tool_call_id="",
                    tool_name=self.name,
                    success=True,
                    output=output,
                    metadata={
                        "pattern": pattern,
                        "matches_found": len(lines),
                        "truncated": len(lines) > max_results,
                    },
                )
            elif result.returncode == 1:
                # No matches found
                return ToolResult(
                    tool_call_id="",
                    tool_name=self.name,
                    success=True,
                    output="No matches found",
                    metadata={"pattern": pattern, "matches_found": 0},
                )
            else:
                return ToolResult(
                    tool_call_id="",
                    tool_name=self.name,
                    success=False,
                    output="",
                    error=f"Grep failed: {result.stderr}",
                )

        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_call_id="",
                tool_name=self.name,
                success=False,
                output="",
                error="Search timed out (> 30 seconds)",
            )
        except Exception as e:
            return ToolResult(
                tool_call_id="",
                tool_name=self.name,
                success=False,
                output="",
                error=f"Search failed: {str(e)}",
            )


class SearchKnowledgeTool(Tool):
    """Search the knowledge base for semantically similar code."""

    def __init__(self, knowledge_store, repo_path: str):
        self.knowledge_store = knowledge_store
        self.repo_path = repo_path

    @property
    def name(self) -> str:
        return "search_knowledge_base"

    @property
    def description(self) -> str:
        return (
            "Search the knowledge base for code similar to a query. "
            "Uses semantic search to find relevant code by meaning, not just keywords. "
            "Useful for finding existing patterns, similar implementations, or related code."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query describing what code to find",
                },
                "module_name": {
                    "type": "string",
                    "description": "Optional: Filter by module name",
                },
                "limit": {
                    "type": "integer",
                    "description": "Optional: Maximum number of results (default: 5)",
                },
            },
            "required": ["query"],
        }

    async def execute(
        self,
        query: str,
        module_name: str | None = None,
        limit: int = 5,
        **kwargs,
    ) -> ToolResult:
        """Search knowledge base."""
        try:
            if not self.knowledge_store:
                return ToolResult(
                    tool_call_id="",
                    tool_name=self.name,
                    success=False,
                    output="",
                    error="Knowledge base not available (repository may not be indexed)",
                )

            # Search
            results = await self.knowledge_store.search_similar(
                query=query,
                limit=limit,
                repo_path=str(self.repo_path),
                module_name=module_name,
            )

            if not results:
                return ToolResult(
                    tool_call_id="",
                    tool_name=self.name,
                    success=True,
                    output="No similar code found",
                    metadata={"query": query, "results_count": 0},
                )

            # Format results
            output_lines = []
            for i, (chunk, score) in enumerate(results, 1):
                output_lines.append(f"Result {i} (score: {score:.3f}):")
                output_lines.append(f"  File: {chunk.file_path}:{chunk.start_line}-{chunk.end_line}")
                output_lines.append(f"  Module: {chunk.module_name}")
                output_lines.append(f"  Type: {chunk.chunk_type}")
                if chunk.name:
                    output_lines.append(f"  Name: {chunk.name}")
                if chunk.purpose:
                    output_lines.append(f"  Purpose: {chunk.purpose}")
                output_lines.append(f"\n{chunk.content}\n")
                output_lines.append("-" * 80)

            return ToolResult(
                tool_call_id="",
                tool_name=self.name,
                success=True,
                output="\n".join(output_lines),
                metadata={
                    "query": query,
                    "results_count": len(results),
                },
            )

        except Exception as e:
            return ToolResult(
                tool_call_id="",
                tool_name=self.name,
                success=False,
                output="",
                error=f"Knowledge base search failed: {str(e)}",
            )
