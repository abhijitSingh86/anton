"""Tools for executing commands and running tests."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from .base import Tool, ToolResult


class RunCommandTool(Tool):
    """Run a CLI command in the repository directory.

    SAFETY: Commands are executed with safety checks and timeout.
    Dangerous commands (rm -rf, etc.) are blocked.
    """

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()

        # Blocked command patterns for safety
        self.blocked_commands = [
            "rm -rf",
            "rm -fr",
            "rmdir",
            "format",
            "mkfs",
            "dd if=",
            "> /dev/",
            "shutdown",
            "reboot",
            "init 0",
            "init 6",
            "curl | sh",
            "wget | sh",
            "eval",
        ]

    @property
    def name(self) -> str:
        return "run_command"

    @property
    def description(self) -> str:
        return (
            "Execute a CLI command in the repository directory. "
            "Use this to run build commands, linters, formatters, or check command output. "
            "Commands run with 60 second timeout. "
            "Dangerous commands (rm -rf, etc.) are blocked for safety."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to execute (e.g., 'npm run build', 'mvn clean install')",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Optional: Timeout in seconds (default: 60, max: 300)",
                },
            },
            "required": ["command"],
        }

    async def execute(
        self,
        command: str,
        timeout: int = 60,
        **kwargs,
    ) -> ToolResult:
        """Execute command."""
        try:
            # Safety check: block dangerous commands
            command_lower = command.lower()
            for blocked in self.blocked_commands:
                if blocked in command_lower:
                    return ToolResult(
                        tool_call_id="",
                        tool_name=self.name,
                        success=False,
                        output="",
                        error=f"Command blocked for safety: contains '{blocked}'",
                    )

            # Enforce timeout limits
            timeout = min(timeout, 300)  # Max 5 minutes

            # Run command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.repo_path),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )

                stdout_text = stdout.decode("utf-8", errors="replace")
                stderr_text = stderr.decode("utf-8", errors="replace")

                # Combine output
                output = ""
                if stdout_text:
                    output += f"STDOUT:\n{stdout_text}\n"
                if stderr_text:
                    output += f"STDERR:\n{stderr_text}\n"

                # Truncate if too long
                if len(output) > 10000:
                    output = output[:10000] + "\n\n... (output truncated)"

                success = process.returncode == 0

                return ToolResult(
                    tool_call_id="",
                    tool_name=self.name,
                    success=success,
                    output=output or "(no output)",
                    error=None if success else f"Command exited with code {process.returncode}",
                    metadata={
                        "command": command,
                        "exit_code": process.returncode,
                    },
                )

            except asyncio.TimeoutError:
                # Kill process if timeout
                try:
                    process.kill()
                    await process.wait()
                except:
                    pass

                return ToolResult(
                    tool_call_id="",
                    tool_name=self.name,
                    success=False,
                    output="",
                    error=f"Command timed out after {timeout} seconds",
                )

        except Exception as e:
            return ToolResult(
                tool_call_id="",
                tool_name=self.name,
                success=False,
                output="",
                error=f"Command execution failed: {str(e)}",
            )


class RunTestsTool(Tool):
    """Run tests for a module or the entire project."""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()

    @property
    def name(self) -> str:
        return "run_tests"

    @property
    def description(self) -> str:
        return (
            "Run tests for a module or the entire project. "
            "Automatically detects test framework (pytest, jest, sbt test, mvn test, etc.) "
            "and runs appropriate command. Use this to validate changes."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "module": {
                    "type": "string",
                    "description": "Optional: Module name to test (omit for all tests)",
                },
                "test_file": {
                    "type": "string",
                    "description": "Optional: Specific test file to run",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Optional: Timeout in seconds (default: 120)",
                },
            },
            "required": [],
        }

    def _detect_test_command(self, module: str | None = None, test_file: str | None = None) -> str | None:
        """Detect appropriate test command based on project type."""
        # Check for various project types
        if (self.repo_path / "build.sbt").exists():
            # Scala/SBT
            if test_file:
                return f'sbt "testOnly {test_file}"'
            elif module:
                return f'sbt "{module}/test"'
            else:
                return "sbt test"

        elif (self.repo_path / "pom.xml").exists():
            # Maven
            if module:
                return f"mvn test -pl :{module}"
            else:
                return "mvn test"

        elif (self.repo_path / "package.json").exists():
            # npm/JavaScript
            if test_file:
                return f"npm test -- {test_file}"
            else:
                return "npm test"

        elif (self.repo_path / "pyproject.toml").exists() or (self.repo_path / "setup.py").exists():
            # Python
            if test_file:
                return f"pytest {test_file}"
            elif module:
                return f"pytest tests/test_{module}.py"
            else:
                return "pytest"

        elif (self.repo_path / "Cargo.toml").exists():
            # Rust
            if module:
                return f"cargo test --package {module}"
            else:
                return "cargo test"

        elif (self.repo_path / "go.mod").exists():
            # Go
            if module:
                return f"go test ./{module}/..."
            else:
                return "go test ./..."

        return None

    async def execute(
        self,
        module: str | None = None,
        test_file: str | None = None,
        timeout: int = 120,
        **kwargs,
    ) -> ToolResult:
        """Run tests."""
        try:
            # Detect test command
            command = self._detect_test_command(module, test_file)

            if not command:
                return ToolResult(
                    tool_call_id="",
                    tool_name=self.name,
                    success=False,
                    output="",
                    error="Could not detect test framework. No build file found (build.sbt, pom.xml, package.json, etc.)",
                )

            # Run tests using run_command tool logic
            run_command_tool = RunCommandTool(str(self.repo_path))
            result = await run_command_tool.execute(command=command, timeout=timeout)

            # Update metadata
            result.tool_name = self.name
            result.metadata.update({
                "module": module,
                "test_file": test_file,
                "test_command": command,
            })

            return result

        except Exception as e:
            return ToolResult(
                tool_call_id="",
                tool_name=self.name,
                success=False,
                output="",
                error=f"Test execution failed: {str(e)}",
            )


class ValidateSyntaxTool(Tool):
    """Validate syntax of code without executing it."""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()

    @property
    def name(self) -> str:
        return "validate_syntax"

    @property
    def description(self) -> str:
        return (
            "Validate syntax of code before applying changes. "
            "Checks for syntax errors without executing the code. "
            "Supports Python, JavaScript, Scala, Java, and more."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Code to validate",
                },
                "language": {
                    "type": "string",
                    "description": "Programming language (python, javascript, scala, java, etc.)",
                },
                "file_path": {
                    "type": "string",
                    "description": "Optional: File path for context (for better error messages)",
                },
            },
            "required": ["code", "language"],
        }

    async def execute(
        self,
        code: str,
        language: str,
        file_path: str | None = None,
        **kwargs,
    ) -> ToolResult:
        """Validate syntax."""
        try:
            language = language.lower()

            if language == "python":
                # Python: use compile()
                try:
                    compile(code, file_path or "<string>", "exec")
                    return ToolResult(
                        tool_call_id="",
                        tool_name=self.name,
                        success=True,
                        output="✓ Python syntax is valid",
                        metadata={"language": language},
                    )
                except SyntaxError as e:
                    return ToolResult(
                        tool_call_id="",
                        tool_name=self.name,
                        success=False,
                        output="",
                        error=f"Python syntax error at line {e.lineno}: {e.msg}",
                        metadata={"language": language, "line": e.lineno},
                    )

            elif language in ("javascript", "js", "typescript", "ts"):
                # JavaScript/TypeScript: use node --check
                result = subprocess.run(
                    ["node", "--check"],
                    input=code,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    return ToolResult(
                        tool_call_id="",
                        tool_name=self.name,
                        success=True,
                        output=f"✓ {language.capitalize()} syntax is valid",
                        metadata={"language": language},
                    )
                else:
                    return ToolResult(
                        tool_call_id="",
                        tool_name=self.name,
                        success=False,
                        output="",
                        error=f"{language.capitalize()} syntax error:\n{result.stderr}",
                        metadata={"language": language},
                    )

            elif language == "scala":
                # Scala: use scalac -Ystop-after:parser
                # This requires scalac to be installed
                import tempfile

                with tempfile.NamedTemporaryFile(mode="w", suffix=".scala", delete=False) as f:
                    f.write(code)
                    temp_file = f.name

                try:
                    result = subprocess.run(
                        ["scalac", "-Ystop-after:parser", temp_file],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if result.returncode == 0:
                        return ToolResult(
                            tool_call_id="",
                            tool_name=self.name,
                            success=True,
                            output="✓ Scala syntax is valid",
                            metadata={"language": language},
                        )
                    else:
                        return ToolResult(
                            tool_call_id="",
                            tool_name=self.name,
                            success=False,
                            output="",
                            error=f"Scala syntax error:\n{result.stderr}",
                            metadata={"language": language},
                        )
                finally:
                    import os
                    os.unlink(temp_file)

            elif language == "java":
                # Java: use javac
                import tempfile

                with tempfile.NamedTemporaryFile(mode="w", suffix=".java", delete=False) as f:
                    f.write(code)
                    temp_file = f.name

                try:
                    result = subprocess.run(
                        ["javac", "-Xdiags:verbose", temp_file],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if result.returncode == 0:
                        return ToolResult(
                            tool_call_id="",
                            tool_name=self.name,
                            success=True,
                            output="✓ Java syntax is valid",
                            metadata={"language": language},
                        )
                    else:
                        return ToolResult(
                            tool_call_id="",
                            tool_name=self.name,
                            success=False,
                            output="",
                            error=f"Java syntax error:\n{result.stderr}",
                            metadata={"language": language},
                        )
                finally:
                    import os
                    os.unlink(temp_file)
                    # Clean up .class file if created
                    class_file = temp_file.replace(".java", ".class")
                    if os.path.exists(class_file):
                        os.unlink(class_file)

            else:
                return ToolResult(
                    tool_call_id="",
                    tool_name=self.name,
                    success=False,
                    output="",
                    error=f"Syntax validation not supported for language: {language}",
                    metadata={"language": language},
                )

        except FileNotFoundError as e:
            return ToolResult(
                tool_call_id="",
                tool_name=self.name,
                success=False,
                output="",
                error=f"Compiler not found: {str(e)}. Install {language} compiler to validate syntax.",
                metadata={"language": language},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_call_id="",
                tool_name=self.name,
                success=False,
                output="",
                error="Syntax validation timed out",
                metadata={"language": language},
            )
        except Exception as e:
            return ToolResult(
                tool_call_id="",
                tool_name=self.name,
                success=False,
                output="",
                error=f"Syntax validation failed: {str(e)}",
                metadata={"language": language},
            )
