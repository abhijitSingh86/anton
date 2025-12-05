"""Tools for Git and dependency analysis."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .base import Tool, ToolResult


class GetGitHistoryTool(Tool):
    """Get Git history for a file to understand how it evolved."""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()

    @property
    def name(self) -> str:
        return "get_git_history"

    @property
    def description(self) -> str:
        return (
            "Get Git commit history for a file. "
            "Shows who changed the file, when, and why (commit messages). "
            "Useful for understanding code evolution and finding related changes."
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
                "max_commits": {
                    "type": "integer",
                    "description": "Optional: Maximum number of commits to show (default: 10)",
                },
            },
            "required": ["file_path"],
        }

    async def execute(
        self,
        file_path: str,
        max_commits: int = 10,
        **kwargs,
    ) -> ToolResult:
        """Get Git history."""
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

            # Run git log
            result = subprocess.run(
                [
                    "git",
                    "log",
                    f"-{max_commits}",
                    "--pretty=format:%h - %an, %ar : %s",
                    "--",
                    file_path,
                ],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                if result.stdout:
                    return ToolResult(
                        tool_call_id="",
                        tool_name=self.name,
                        success=True,
                        output=result.stdout,
                        metadata={
                            "file_path": file_path,
                            "commits_shown": len(result.stdout.strip().split("\n")),
                        },
                    )
                else:
                    return ToolResult(
                        tool_call_id="",
                        tool_name=self.name,
                        success=True,
                        output="No commit history found for this file",
                        metadata={"file_path": file_path, "commits_shown": 0},
                    )
            else:
                # Check if it's because file doesn't exist
                if "does not exist" in result.stderr.lower() or "no such file" in result.stderr.lower():
                    return ToolResult(
                        tool_call_id="",
                        tool_name=self.name,
                        success=False,
                        output="",
                        error=f"File not found: {file_path}",
                    )
                else:
                    return ToolResult(
                        tool_call_id="",
                        tool_name=self.name,
                        success=False,
                        output="",
                        error=f"Git log failed: {result.stderr}",
                    )

        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_call_id="",
                tool_name=self.name,
                success=False,
                output="",
                error="Git log timed out (> 10 seconds)",
            )
        except Exception as e:
            return ToolResult(
                tool_call_id="",
                tool_name=self.name,
                success=False,
                output="",
                error=f"Git history lookup failed: {str(e)}",
            )


class GetDependenciesTool(Tool):
    """Get dependency information for a module or file."""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()

    @property
    def name(self) -> str:
        return "get_dependencies"

    @property
    def description(self) -> str:
        return (
            "Get dependency information for a module or file. "
            "Shows which modules/libraries this module depends on. "
            "Useful for understanding module relationships and impacts of changes."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "module": {
                    "type": "string",
                    "description": "Module name to analyze",
                },
            },
            "required": ["module"],
        }

    def _get_sbt_dependencies(self, module: str) -> str | None:
        """Get dependencies from SBT build file."""
        try:
            # Look for build.sbt or project-specific build file
            build_files = [
                self.repo_path / "build.sbt",
                self.repo_path / module / "build.sbt",
                self.repo_path / module / "build.sbt.scala",
            ]

            for build_file in build_files:
                if build_file.exists():
                    with open(build_file, "r") as f:
                        content = f.read()

                    # Extract libraryDependencies
                    lines = []
                    in_deps = False
                    for line in content.split("\n"):
                        if "libraryDependencies" in line:
                            in_deps = True
                        if in_deps:
                            lines.append(line)
                            if line.strip().endswith(")"):
                                in_deps = False

                    if lines:
                        return "\n".join(lines)

            return None
        except Exception:
            return None

    def _get_maven_dependencies(self, module: str) -> str | None:
        """Get dependencies from Maven pom.xml."""
        try:
            pom_files = [
                self.repo_path / "pom.xml",
                self.repo_path / module / "pom.xml",
            ]

            for pom_file in pom_files:
                if pom_file.exists():
                    # Run mvn dependency:tree
                    result = subprocess.run(
                        ["mvn", "dependency:tree", "-pl", module],
                        cwd=str(self.repo_path),
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode == 0:
                        return result.stdout
            return None
        except Exception:
            return None

    def _get_npm_dependencies(self, module: str) -> str | None:
        """Get dependencies from package.json."""
        try:
            package_files = [
                self.repo_path / "package.json",
                self.repo_path / module / "package.json",
            ]

            for package_file in package_files:
                if package_file.exists():
                    import json

                    with open(package_file, "r") as f:
                        data = json.load(f)

                    deps = []
                    if "dependencies" in data:
                        deps.append("Dependencies:")
                        for name, version in data["dependencies"].items():
                            deps.append(f"  {name}: {version}")

                    if "devDependencies" in data:
                        deps.append("\nDev Dependencies:")
                        for name, version in data["devDependencies"].items():
                            deps.append(f"  {name}: {version}")

                    if deps:
                        return "\n".join(deps)

            return None
        except Exception:
            return None

    def _get_python_dependencies(self, module: str) -> str | None:
        """Get dependencies from requirements.txt or pyproject.toml."""
        try:
            # Check requirements.txt
            req_files = [
                self.repo_path / "requirements.txt",
                self.repo_path / module / "requirements.txt",
            ]

            for req_file in req_files:
                if req_file.exists():
                    with open(req_file, "r") as f:
                        return f.read()

            # Check pyproject.toml
            toml_files = [
                self.repo_path / "pyproject.toml",
                self.repo_path / module / "pyproject.toml",
            ]

            for toml_file in toml_files:
                if toml_file.exists():
                    with open(toml_file, "r") as f:
                        content = f.read()

                    # Extract dependencies section
                    lines = []
                    in_deps = False
                    for line in content.split("\n"):
                        if "[tool.poetry.dependencies]" in line or "[project.dependencies]" in line:
                            in_deps = True
                        elif in_deps and line.startswith("["):
                            in_deps = False
                        elif in_deps:
                            lines.append(line)

                    if lines:
                        return "\n".join(lines)

            return None
        except Exception:
            return None

    async def execute(
        self,
        module: str,
        **kwargs,
    ) -> ToolResult:
        """Get dependencies."""
        try:
            # Try different build systems
            dependencies = (
                self._get_sbt_dependencies(module)
                or self._get_maven_dependencies(module)
                or self._get_npm_dependencies(module)
                or self._get_python_dependencies(module)
            )

            if dependencies:
                return ToolResult(
                    tool_call_id="",
                    tool_name=self.name,
                    success=True,
                    output=dependencies,
                    metadata={"module": module},
                )
            else:
                return ToolResult(
                    tool_call_id="",
                    tool_name=self.name,
                    success=False,
                    output="",
                    error=f"Could not find dependencies for module '{module}'. No build file found.",
                )

        except Exception as e:
            return ToolResult(
                tool_call_id="",
                tool_name=self.name,
                success=False,
                output="",
                error=f"Dependency lookup failed: {str(e)}",
            )
