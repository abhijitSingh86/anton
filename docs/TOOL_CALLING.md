# Tool Calling System

Anton now includes a comprehensive tool calling system that enables agents to invoke external tools and functions during task execution. This makes agents much more intelligent and autonomous.

## ✨ Features

- **🔍 Code Search**: Read files, search codebase, query knowledge base
- **⚙️ Command Execution**: Run CLI commands, tests, and validation
- **📊 Analysis**: Get dependencies, Git history, and project info
- **🛡️ Safety**: Built-in command blocking and sandboxing
- **🔌 Provider Support**: Works with Claude and OpenAI function calling APIs

## 🎯 Available Tools

### Code Tools

#### `read_file`
Read contents of a file in the repository.

```python
{
  "file_path": "src/main/api/UserController.scala",
  "start_line": 10,  # Optional
  "end_line": 50      # Optional
}
```

**Use cases**:
- Examine code before making changes
- Understand implementation details
- Check existing patterns

#### `grep_codebase`
Search for patterns in the codebase using grep.

```python
{
  "pattern": "class UserController",
  "file_pattern": "*.scala",  # Optional
  "ignore_case": true,         # Optional
  "max_results": 20            # Optional (default: 50)
}
```

**Use cases**:
- Find where functions/classes are defined
- Find all usages of a method
- Search for specific patterns

#### `search_knowledge_base`
Search the knowledge base for semantically similar code.

```python
{
  "query": "user authentication logic",
  "module_name": "api",  # Optional
  "limit": 5             # Optional (default: 5)
}
```

**Use cases**:
- Find existing patterns to replicate
- Discover similar implementations
- Learn from existing code

### Execution Tools

#### `run_command`
Execute a CLI command in the repository directory.

```python
{
  "command": "npm run build",
  "timeout": 60  # Optional (default: 60, max: 300)
}
```

**Safety**: Dangerous commands (rm -rf, etc.) are automatically blocked.

**Use cases**:
- Run build commands
- Check lint/format output
- Execute any safe CLI command

#### `run_tests`
Run tests for a module or the entire project.

```python
{
  "module": "api",          # Optional
  "test_file": "test_user", # Optional
  "timeout": 120            # Optional (default: 120)
}
```

**Auto-detects**: pytest, jest, sbt test, mvn test, cargo test, go test

**Use cases**:
- Validate changes before proposing them
- Check if tests pass
- Run specific test files

#### `validate_syntax`
Validate syntax of code without executing it.

```python
{
  "code": "def hello():\n    print('world')",
  "language": "python",
  "file_path": "src/utils.py"  # Optional
}
```

**Supported**: Python, JavaScript, TypeScript, Scala, Java

**Use cases**:
- Check syntax before proposing changes
- Validate generated code
- Catch errors early

### Analysis Tools

#### `get_git_history`
Get Git commit history for a file.

```python
{
  "file_path": "src/main/api/UserController.scala",
  "max_commits": 10  # Optional (default: 10)
}
```

**Use cases**:
- Understand how code evolved
- See who made changes and why
- Find related changes

#### `get_dependencies`
Get dependency information for a module.

```python
{
  "module": "api"
}
```

**Auto-detects**: build.sbt, pom.xml, package.json, requirements.txt, pyproject.toml

**Use cases**:
- Understand module dependencies
- Check if library is available
- Assess impact of dependency changes

## 🚀 Quick Start

### For Users

Tool calling is **enabled by default** when available. Agents automatically use tools when needed.

```bash
# Tools are used automatically
anton task . "Add caching to UserService"

# Agent will:
# 1. read_file to examine UserService
# 2. search_knowledge_base for caching patterns
# 3. validate_syntax before proposing changes
# 4. run_tests to validate implementation
```

### Enable/Disable

```bash
# Enable explicitly (default)
anton task . "task" --enable-tools

# Disable tools
anton task . "task" --no-tools

# Disable command execution only (read-only mode)
anton task . "task" --no-command-execution
```

## 🛠️ How It Works

### Tool Call Flow

```
1. Agent sends request to LLM with tools available
2. LLM decides to call a tool (e.g., read_file)
3. Tool is executed safely in sandboxed environment
4. Result returned to LLM
5. LLM continues with tool output as context
6. Process repeats until task is complete
```

### Example Conversation

```
User: "Add input validation to UserController"

Agent → LLM: What tools do you need?
LLM → Tool Call: read_file("src/api/UserController.scala")
Tool → LLM: [file contents]

LLM → Tool Call: search_knowledge_base("input validation patterns")
Tool → LLM: [similar code examples]

LLM → Tool Call: validate_syntax(proposed_code, "scala")
Tool → LLM: ✓ Syntax valid

LLM → Tool Call: run_tests(module="api")
Tool → LLM: ✓ All tests passed

LLM → Agent: Here's the implementation [with validated changes]
```

## 🛡️ Safety & Security

### Command Execution Safety

**Blocked commands** (automatically):
- `rm -rf` - Recursive deletion
- `format` - Disk formatting
- `dd if=` - Disk operations
- `shutdown` / `reboot`
- `curl | sh` - Piped execution
- `eval` - Code evaluation

**Timeouts**:
- Commands: 60 seconds (max 300)
- Tests: 120 seconds
- Searches: 30 seconds

**Sandboxing**:
- Commands run in repository directory only
- File access restricted to repository
- No access to parent directories
- Environment isolation

### File Access Safety

**read_file** security:
- Path must be within repository
- No access to parent directories (`../` blocked)
- No access to system files

**grep_codebase** exclusions:
- `.git/` - Version control
- `node_modules/` - Dependencies
- `venv/` - Virtual environments
- `__pycache__/` - Python cache
- `.modular-agents/` - Anton runtime

## 📊 Tool Usage Examples

### Example 1: Intelligent Code Search

```python
# Agent needs to add caching
# 1. Search for existing patterns
search_knowledge_base("caching implementation")

# 2. Read found example
read_file("src/services/ProductService.scala", start_line=45, end_line=75)

# 3. Adapt pattern and validate
validate_syntax(adapted_code, "scala")
```

### Example 2: Test-Driven Development

```python
# Agent implements new feature
# 1. Read existing tests
read_file("tests/test_user_service.py")

# 2. Implement feature
# [code generation]

# 3. Validate syntax
validate_syntax(new_code, "python")

# 4. Run tests
run_tests(test_file="tests/test_user_service.py")

# 5. If tests fail, iterate
```

### Example 3: Dependency Management

```python
# Agent needs to use a library
# 1. Check if dependency exists
get_dependencies("api")

# 2. If missing, search for usage examples
grep_codebase("import redis", file_pattern="*.py")

# 3. Validate it works
run_command("pip list | grep redis")
```

## 🔧 Configuration

### CLI Flags

```bash
# Enable tools (default)
anton task . "description" --enable-tools

# Disable all tools
anton task . "description" --no-tools

# Enable read-only tools (no commands)
anton task . "description" --no-command-execution

# Custom tool timeout
anton task . "description" --tool-timeout 300

# Knowledge base tools
anton task . "description" --use-knowledge  # Enables search_knowledge_base
```

### Programmatic Usage

```python
from modular_agents import AgentRuntime, LLMConfig
from modular_agents.tools import register_default_tools

# Create runtime with tools
runtime = AgentRuntime(
    repo_path="/path/to/repo",
    llm_provider="claude",
    llm_config=LLMConfig(model="claude-sonnet-4", api_key="..."),
)

# Register tools
tools = register_default_tools(
    repo_path="/path/to/repo",
    knowledge_store=knowledge_store,  # Optional
    allow_commands=True,  # Enable command execution
)

# Tools are automatically used by agents
result = await runtime.execute_task("Add feature")
```

## 📈 Performance Impact

### Tool Call Overhead

- **Tool execution**: 10ms - 10s (depends on tool)
- **LLM latency**: +100-500ms per tool call
- **Network**: Additional API roundtrips

### Typical Tool Usage

**Simple task** (refactor function):
- 2-3 `read_file` calls
- 1 `validate_syntax` call
- Total: ~2-3 seconds overhead

**Complex task** (add feature):
- 5-10 `read_file` calls
- 2-3 `search_knowledge_base` calls
- 2-3 `run_tests` calls
- Total: ~15-30 seconds overhead

**Benefits outweigh costs**:
- ✅ Higher quality code (validated before proposing)
- ✅ Fewer errors (syntax checked, tests run)
- ✅ Better context (searches existing patterns)
- ✅ More autonomous (can validate own work)

## 🎓 Best Practices

### For Agents

1. **Read before writing**: Always `read_file` before modifying
2. **Search for patterns**: Use `search_knowledge_base` to find existing approaches
3. **Validate early**: Check `validate_syntax` before proposing large changes
4. **Test frequently**: Run `run_tests` after implementation
5. **Check dependencies**: Use `get_dependencies` before adding imports

### For Users

1. **Enable knowledge base**: Index repository for better `search_knowledge_base` results
2. **Trust validation**: Agents can run tests to verify their work
3. **Review tool calls**: Check what tools were used (in verbose mode)
4. **Set appropriate timeouts**: Increase for large codebases
5. **Use read-only mode**: Disable commands for untrusted tasks

## 🔍 Debugging

### Verbose Mode

```bash
# See all tool calls
anton task . "description" --verbose

# Output:
# [Tool Call] read_file(file_path="src/api/User.scala")
# [Tool Result] Success: 150 lines read
# [Tool Call] validate_syntax(code="...", language="scala")
# [Tool Result] Success: ✓ Scala syntax is valid
```

### Check Tool Availability

```python
from modular_agents.tools import register_default_tools

tools = register_default_tools("/path/to/repo")
print(f"Available tools: {[t.name for t in tools.list_tools()]}")

# Output:
# Available tools: ['read_file', 'grep_codebase', 'search_knowledge_base',
#                   'run_command', 'run_tests', 'validate_syntax',
#                   'get_dependencies', 'get_git_history']
```

## 🚧 Limitations

### Current Limitations

1. **No interactive commands**: Tools can't handle prompts (stdin)
2. **Output size**: Command output truncated at 10KB
3. **No real-time output**: Can't see progress during execution
4. **Compiler availability**: Syntax validation requires compilers installed
5. **Git repository**: Some tools require Git to be initialized

### Future Enhancements

- [ ] Add `write_file` tool (with approval workflow)
- [ ] Add `run_interactive` for commands needing input
- [ ] Add `stream_output` for real-time command output
- [ ] Add `debug_code` tool for step-by-step execution
- [ ] Add `refactor_tool` for safe automated refactoring
- [ ] Add `benchmark_code` for performance testing
- [ ] Add `security_scan` for vulnerability checking

## 📚 Implementation Status

### ✅ Implemented

- [x] Tool base infrastructure
- [x] Tool registry system
- [x] All 8 core tools
- [x] Safety checks and sandboxing
- [x] Command blocking
- [x] Timeout enforcement
- [x] Error handling
- [x] Tool result formatting

### 🔄 In Progress

- [ ] LLM provider updates (Claude, OpenAI)
- [ ] Agent integration (OrchestratorAgent, ModuleAgent)
- [ ] CLI configuration
- [ ] Tool call logging and telemetry

### 📋 Planned

- [ ] Tool usage analytics
- [ ] Custom tool plugins
- [ ] Tool caching
- [ ] Parallel tool execution
- [ ] Tool result validation

## 🤝 Contributing

### Adding New Tools

Create a new tool by subclassing `Tool`:

```python
from modular_agents.tools.base import Tool, ToolResult

class MyCustomTool(Tool):
    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "Description of what my tool does"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "..."},
            },
            "required": ["param1"],
        }

    async def execute(self, param1: str, **kwargs) -> ToolResult:
        # Implement tool logic
        return ToolResult(
            tool_call_id="",
            tool_name=self.name,
            success=True,
            output="Result",
        )
```

Then register it:

```python
from modular_agents.tools import ToolRegistry

registry = ToolRegistry()
registry.register(MyCustomTool())
```

## 📖 See Also

- [Agent Continuation](CONTINUATION.md) - Checkpointing and retry
- [Knowledge Base](KNOWLEDGE_BASE.md) - Semantic code search
- [Development Guide](DEVELOPMENT.md) - Contributing to Anton

---

**Tool calling makes Anton agents intelligent, autonomous, and self-validating!** 🚀
