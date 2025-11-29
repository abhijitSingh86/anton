# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**anton** (aka "Modular Agents") is a multi-agent framework for modular codebases. It analyzes project structure, creates specialized LLM agents for each module, and coordinates task execution across modules.

Key architecture:
- **OrchestratorAgent**: Decomposes tasks, creates execution plans (topological sort of subtasks), delegates to module agents
- **ModuleAgent**: Specialized agent per module with knowledge of that module's code, dependencies, and API
- **Analyzers**: Pluggable analyzers for different project types (SBT, Maven, npm, etc.)
- **LLM Providers**: Pluggable backends (Claude, OpenAI, Ollama, OpenAI-compatible)

## Development Commands

### Installation
```bash
# Development mode with all providers
pip install -e ".[all,dev]"

# Specific providers
pip install -e ".[claude]"
pip install -e ".[openai]"
pip install -e ".[ollama]"
```

### Testing
```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_core.py::TestModels::test_module_profile_creation

# Run with verbose output
pytest -v

# Run async tests (already configured via pytest.ini_options)
pytest tests/
```

### Linting
```bash
# Check code
ruff check src/

# Fix auto-fixable issues
ruff check --fix src/
```

### CLI Usage

#### Repository Initialization
```bash
# Initialize repository - analyze and save knowledge
anton init /path/to/repo

# Initialize with interactive enrichment (add custom descriptions)
anton init /path/to/repo --enrich

# Force re-analysis even if knowledge exists
anton init /path/to/repo --force

# Initialize and export to markdown
anton init /path/to/repo --export knowledge.md

# Enrich existing knowledge without re-analyzing
anton enrich /path/to/repo

# Export knowledge to markdown
anton export /path/to/repo --output knowledge.md
```

#### Running Tasks
```bash
# Analyze repository structure (no LLM required)
anton analyze /path/to/repo
anton analyze /path/to/repo --json

# Interactive agent loop (requires API key)
export ANTHROPIC_API_KEY="your-key"
anton run /path/to/repo --provider claude

# Single task execution
anton task /path/to/repo "Add caching to UserRepository" --provider claude

# Dry run (show plan without executing)
anton task /path/to/repo "task description" --dry-run

# Verbose mode - shows LLM interactions in real-time
anton task /path/to/repo "task description" --verbose

# Debug mode - shows full prompts, responses, and saves traces
anton task /path/to/repo "task description" --debug

# List available providers
anton providers
```

## Knowledge Management & Initialization

The framework includes a comprehensive knowledge management system for caching repository analysis and enriching it with custom information.

### How Knowledge Works

1. **Auto-Analysis**: First time, analyzes repository structure
2. **Caching**: Saves to `.modular-agents/knowledge.json`
3. **Custom Enrichment**: Optionally add user-provided metadata in `.modular-agents/custom.json`
4. **Hierarchical Loading**: Loads from current directory or parent directories (for submodules)

### Initialization Workflow

```bash
# Step 1: Initialize repository (first time)
anton init .

# The system:
# 1. Analyzes project structure (build files, source code)
# 2. Discovers modules and their relationships
# 3. Infers module purposes from code
# 4. Saves to .modular-agents/knowledge.json

# Step 2: Enrich with custom information (optional)
anton enrich .

# Interactively add:
# - Custom module descriptions
# - Tags and categories
# - Architecture notes
# - Team conventions

# Step 3: Use for agent tasks
anton task . "Implement feature X"
# Agents load cached knowledge + custom enrichments
```

### Knowledge Files Structure

```
.modular-agents/
├── knowledge.json           # Auto-analyzed repository structure
├── custom.json             # User-provided enrichments
├── summary_<task_id>.md    # Task execution summaries
└── traces/                 # LLM interaction logs
    ├── orchestrator_*.jsonl
    ├── module_*_*.jsonl
    └── trace_summary.json
```

### Custom Metadata Format

`.modular-agents/custom.json`:
```json
{
  "description": "E-commerce platform with microservices architecture",
  "architecture_notes": "Uses event sourcing; all services communicate via Kafka",
  "modules": {
    "api": {
      "purpose": "Public REST API - rate limited, versioned endpoints",
      "tags": ["public", "critical", "monitored"],
      "notes": "Must maintain backward compatibility"
    },
    "domain": {
      "purpose": "Core business logic - DDD patterns",
      "tags": ["core", "well-tested"]
    }
  }
}
```

### Hierarchical Knowledge Loading

The system searches for knowledge in this order:
1. Current directory: `.modular-agents/knowledge.json`
2. Parent directory (up to 3 levels up)
3. Custom metadata from current directory always takes precedence

**Use Case: Submodules**
```
/project/                    # Main repo with anton init
├── .modular-agents/
│   └── knowledge.json       # Main repo knowledge
└── services/
    └── api-service/         # Submodule
        └── .modular-agents/
            └── custom.json  # Submodule-specific enrichments

# Running from api-service/
anton task . "Add endpoint"
# Loads: parent knowledge + local custom metadata
```

### Enrichment Features

When running `anton init --enrich` or `anton enrich`, you can customize:

**Per-Module:**
- Custom purpose/description (overrides auto-inferred)
- Tags (e.g., "critical", "legacy", "api", "internal")
- Additional notes for agents

**Repository-Level:**
- Overall description
- Architecture notes
- Conventions and constraints

### Export Knowledge

```bash
# Export to markdown for documentation
anton export . --output docs/architecture.md

# Generates human-readable documentation:
# - Module purposes and relationships
# - Dependency graph
# - File counts and structure
```

### Best Practices

1. **Initialize once per repository**: Run `anton init` at the root
2. **Enrich with team knowledge**: Add conventions, constraints, critical info
3. **Use tags for categorization**: "api", "public", "critical", "legacy"
4. **Document architecture patterns**: Event sourcing, DDD, microservices style
5. **Update after major refactors**: Use `anton init --force` to re-analyze
6. **Export for onboarding**: Generate markdown docs for new team members

## Architecture Details

### Core Data Flow
```
User Task → OrchestratorAgent.process_task()
  ↓
1. decompose_task() - LLM generates subtasks with module assignments
  ↓
2. create_execution_plan() - Topological sort into parallelizable phases
  ↓
3. execute phases - Each phase runs subtasks concurrently via asyncio.gather()
  ↓
4. ModuleAgent.execute_task() - Implements changes within module boundary
  ↓
5. Generate summary - Orchestrator summarizes all results
```

### Key Components

**`src/modular_agents/core/models.py`**
- All Pydantic models: `ModuleProfile`, `RepoKnowledge`, `Task`, `SubTask`, `TaskResult`, etc.
- Task execution uses phases (`ExecutionPlan`, `ExecutionPhase`) for parallel work
- Dependencies tracked via `SubTask.depends_on` field

**`src/modular_agents/runtime.py`**
- `AgentRuntime`: Main entry point, manages initialization and task execution
- `InteractiveLoop`: REPL for interactive mode
- Caches repo analysis in `.modular-agents/knowledge.json`
- Stores module mapping for agent reporting

**`src/modular_agents/reporting.py`**
- `AgentSummaryReporter`: Generates visual summaries and reports
- `display_agent_summary()`: Rich table view of agent activities
- `display_detailed_agent_learning()`: Tree view with full details
- `save_learning_summary()`: Saves markdown report to disk

**`src/modular_agents/knowledge.py`**
- `KnowledgeManager`: Manages knowledge persistence and enrichment
- `save_knowledge()` / `load_knowledge()`: JSON serialization
- `load_from_hierarchy()`: Searches current and parent directories
- `enrich_knowledge()`: Applies custom metadata overlays
- `interactive_enrich()`: Interactive prompts for customization
- `export_knowledge()`: Generates markdown documentation

**`src/modular_agents/agents/orchestrator.py`**
- `OrchestratorAgent.decompose_task()`: LLM call to generate subtasks (expects JSON response)
- `OrchestratorAgent.create_execution_plan()`: Topological sort for parallel execution
- Uses `asyncio.gather()` for concurrent subtask execution within phases

**`src/modular_agents/agents/module_agent.py`**
- Each module gets own agent with specialized system prompt
- Must respect module boundaries (report blockers for cross-module changes)
- Should follow existing patterns in the module

**`src/modular_agents/analyzers/`**
- `BaseAnalyzer`: Abstract base class with `can_analyze()` and `analyze()` methods
- `SBTAnalyzer`: Parses `build.sbt`, discovers modules via regex, infers purpose
- `GenericAnalyzer`: Fallback using LLM-based analysis
- Registry pattern: `@AnalyzerRegistry.register` decorator auto-registers analyzers

**`src/modular_agents/llm/`**
- `LLMProvider`: Abstract base for all LLM backends
- `ClaudeProvider`, `OpenAIProvider`, `OllamaProvider`: Concrete implementations
- Registry pattern: `LLMProviderRegistry` manages provider lookup

### Important Patterns

1. **Registry Pattern**: Both analyzers and LLM providers use decorator-based registration
   ```python
   @AnalyzerRegistry.register
   class MyAnalyzer(BaseAnalyzer):
       ...
   ```

2. **Async-First**: All analysis and agent operations are async (`async def`, `await`, `asyncio.gather`)

3. **Path Handling**: Models use `Path` objects but JSON serialization requires string conversion (see `runtime.py:_save_knowledge()`)

4. **Pydantic Models**: All data structures use Pydantic v2 with `BaseModel` for validation and serialization

5. **Agent Communication**: Agents use LLM's `chat()` method; orchestrator parses JSON from LLM responses

6. **Empty Module Handling**: Module agents can bootstrap empty modules (0 files) by creating initial files. The system detects if an agent incorrectly reports a blocker for an empty module and provides actionable error messages.

7. **Language Enforcement**: Module agents MUST use the correct language for each module. The system:
   - Detects language from file extensions during analysis
   - Includes language in module profiles
   - Enforces language in system prompts (with explicit warnings)
   - Validates file extensions in responses
   - Fails tasks that create files in wrong language

8. **Style Matching**: Agents receive code examples from existing files and must match:
   - Exact code style and formatting
   - Naming conventions (PascalCase, camelCase, etc.)
   - Framework patterns (case classes, traits, etc.)
   - Project-specific conventions

## Adding New Analyzers

To support a new project type:

1. Create `src/modular_agents/analyzers/mytype.py`
2. Subclass `BaseAnalyzer` and implement:
   - `project_type` property
   - `can_analyze(path: Path) -> bool`
   - `async analyze(path: Path) -> RepoKnowledge`
3. Use `@AnalyzerRegistry.register` decorator
4. Import in `src/modular_agents/analyzers/__init__.py`

See `sbt.py` for reference implementation with regex-based parsing.

## Adding New LLM Providers

1. Create `src/modular_agents/llm/myprovider.py`
2. Subclass `LLMProvider` and implement:
   - `__init__(config: LLMConfig)`
   - `async chat(messages: list[dict], system: str) -> str`
3. Use `@LLMProviderRegistry.register("provider-name")` decorator
4. Import in `src/modular_agents/llm/__init__.py`
5. Add to optional dependencies in `pyproject.toml`

## Environment Variables

- `ANTHROPIC_API_KEY`: For Claude provider
- `OPENAI_API_KEY`: For OpenAI provider
- `LLM_API_KEY`: Generic fallback if provider-specific key not set

## Agent Reporting and Summaries

The framework provides comprehensive reporting of what each agent learned and accomplished during task execution.

### Agent Summary Reports

After each task, the system automatically displays:

1. **Summary Table** - Shows each module agent's activity:
   - Agent name (by module)
   - Status (✓ Completed, ✗ Failed, ⊗ Blocked)
   - Files changed count
   - Tests added count
   - Brief summary of work

2. **Detailed Tree View** - Hierarchical view of agent learning:
   - Each agent's subtasks
   - Files created/modified/deleted (with action types color-coded)
   - Tests added
   - Implementation notes
   - Errors encountered
   - Blockers reported

3. **Markdown Summary** - Saved to `.modular-agents/summary_<task_id>.md`:
   - Complete record of all agent activities
   - Formatted for easy review
   - Includes statistics and success rates

### Generated Files

All agent artifacts are stored in `.modular-agents/`:
- `knowledge.json` - Cached repository analysis
- `summary_<task_id>.md` - Task execution summary (one per task)
- `traces/` - LLM interaction logs (when using --verbose or --debug)
- `traces/trace_summary.json` - Summary of all LLM calls

### Agent Identification

Agents are identified by module name:
- Module agents: `module_<name>` (displayed as 🤖 Agent[name])
- Orchestrator: `orchestrator` (displayed as 🎯 Orchestrator)

In verbose/debug output:
```
🎯 Orchestrator → claude/claude-sonnet-4
   (decomposing task into subtasks)

🤖 Agent[api] → claude/claude-sonnet-4
   (implementing API endpoints)

🤖 Agent[domain] → claude/claude-sonnet-4
   (updating domain models)
```

### Example Summary Output

```
┌─ Agent Learning Summary ─────────────────────────────────┐
│ Module Agent Activities                                  │
├──────────────┬────────────┬────────┬───────┬─────────────┤
│ Agent        │ Status     │ Files  │ Tests │ Summary     │
├──────────────┼────────────┼────────┼───────┼─────────────┤
│ api          │ ✓ Completed│   3    │   2   │ Added new..│
│ domain       │ ✓ Completed│   1    │   1   │ Updated... │
│ persistence  │ ✗ Failed   │   0    │   0   │ JSON err...│
└──────────────┴────────────┴────────┴───────┴─────────────┘

Overall Statistics:
  Total files changed: 4
  Total tests added: 3
  Modules involved: 3
  Success rate: 2/3 subtasks
```

## Debugging and Tracing

The framework includes comprehensive tracing capabilities to debug agent behavior and LLM interactions.

### Tracing Infrastructure

**`src/modular_agents/trace.py`**
- `TraceLogger`: Centralized logging of all LLM interactions
- `LLMInteraction`: Records each LLM call with full context (prompts, responses, timing, errors)
- Automatically integrated into `BaseAgent.think()` method

### Using Tracing

**CLI Flags:**
- `--verbose` / `-v`: Show LLM interactions in real-time (with JSON highlighting)
- `--debug` / `-d`: Show full debug output including complete prompts and system messages

**Trace Files:**
When `--verbose` or `--debug` is used, traces are saved to:
- `.modular-agents/traces/<agent_name>_<timestamp>.jsonl` - One line per interaction (JSONL format)
- `.modular-agents/traces/trace_summary.json` - Summary of all interactions at end of task

**Example:**
```bash
# Debug a failing task
anton task . "task description" --debug

# Check the traces
cat .modular-agents/traces/trace_summary.json
cat .modular-agents/traces/module_api_*.jsonl
```

### Error Reporting

**Module Agents** provide detailed error information when JSON parsing fails:
- Shows exact location of JSON decode errors (line/column)
- Includes raw LLM response in error notes
- Attempts to clean common JSON issues (trailing commas, control characters)
- Infers task status from response text when JSON parsing fails

**Orchestrator** provides detailed debugging when task decomposition fails:
- Shows whether JSON was found in response
- Displays syntax-highlighted JSON with line numbers
- Validates response structure (missing fields, wrong types)
- Provides specific error messages for each failure mode
- Suggests fixes: rephrasing task, checking modules, using --debug

**Common Errors and Solutions:**

| Error | Cause | Solution |
|-------|-------|----------|
| "Could not decompose task" | LLM didn't return valid JSON | Run with `--debug` to see response |
| "No JSON found in response" | LLM returned text instead of JSON | Rephrase task or check LLM config |
| "Missing 'subtasks' field" | Wrong JSON structure | Check trace logs |
| "Empty subtasks list" | No modules match task | Verify with `anton analyze .` |
| "Subtask missing required fields" | Incomplete JSON | Check --debug output |

### Trace Data Structure

Each `LLMInteraction` contains:
- `timestamp`: When the call was made
- `agent_name`: Which agent made the call
- `provider` / `model`: LLM backend used
- `system_prompt`: Full system prompt
- `messages`: Conversation history
- `response`: LLM response text
- `duration_ms`: Call duration
- `error`: Error message if call failed

## Language & Style Enforcement

The framework includes strict enforcement to prevent agents from creating code in wrong languages or incompatible styles.

### Language Detection

**Automatic Detection:**
- SBT projects → Scala
- Maven/Gradle with `.java` → Java
- `package.json` → JavaScript/TypeScript
- `setup.py`/`pyproject.toml` → Python

**Module Profiles Include:**
```python
ModuleProfile(
    name="api",
    language="scala",          # Detected from files
    framework="Akka",          # Detected from dependencies
    code_examples=[...],       # Extracted from existing code
    naming_patterns=[...],     # Detected conventions
)
```

### Agent Enforcement

**Module Agent System Prompts:**
```
## CRITICAL RULES

### 1. LANGUAGE ENFORCEMENT
**YOU MUST ONLY USE SCALA**
- Do NOT create Python files (.py) if this is a scala module
- Do NOT create Java files (.java) if this is a scala module
- ONLY create scala files with .scala extension

### 2. STYLE MATCHING (MANDATORY)
- Study the code examples above
- Match the EXACT style, indentation, formatting
- Use the SAME naming conventions
```

**Validation:**
- Checks file extensions in agent responses
- Fails if wrong language detected
- Shows clear error: "LANGUAGE VIOLATION: Expected scala files ['.scala'], got wrong files: ['file.py']"

### Code Examples Provided to Agents

Agents receive actual code from the module:
```scala
// From UserService.scala
class UserService(repository: UserRepository) {
  def findById(id: UserId): Future[Option[User]] = {
    repository.find(id)
  }
}
```

This ensures agents:
- See actual style
- Understand patterns
- Match formatting
- Use correct imports

### Naming Pattern Detection

Auto-detected patterns:
- "Classes use PascalCase"
- "Methods use camelCase"
- "Uses case classes for data models"
- "Uses implicit parameters/conversions"
- "Uses trait-based composition"

### Framework Detection

From dependencies:
- Akka → Use Akka patterns (actors, futures)
- Play Framework → Use Play conventions
- ZIO → Use ZIO effects
- Spring → Use Spring annotations

### Error Messages

When agents violate language rules:
```
LANGUAGE VIOLATION: Agent created files in wrong language!
Expected: scala files ['.scala']
Got wrong files: src/main/domain.py, src/test/test_domain.py

The module 'domain' is a scala module.
All code files MUST use scala.
This is a critical error that violates repository consistency.
```

### Best Practices

1. **Run init to detect language**: `anton init .` analyzes and caches language info
2. **Check module language**: `anton analyze .` shows detected language
3. **Enrich with style notes**: `anton enrich .` to add custom conventions
4. **Review generated code**: Always check language matches

## Configuration

- **Package name**: `anton` (PyPI) / `modular_agents` (Python import)
- **Entry point**: `anton` command → `modular_agents.cli:main`
- **Python**: >=3.10 required
- **Build system**: Hatchling (PEP 517)
- **Code style**: Ruff with line length 100
- **Test framework**: pytest with asyncio support enabled
