# Anton (Modular Agents)

A production-ready multi-agent framework for autonomous code development in modular codebases. Anton analyzes your project structure, creates specialized AI agents for each module, and coordinates complex tasks with automatic checkpointing, retry logic, and fault tolerance.

## ✨ Key Features

- **🤖 Autonomous Development**: Long-running tasks with automatic checkpointing and resume
- **🔧 Tool Calling**: Agents can read files, search code, run tests, and validate syntax autonomously
- **🔄 Fault Tolerant**: Automatic retry with exponential backoff for transient failures
- **📊 Real-Time Progress**: Track task completion and monitor live progress
- **🛡️ Safety First**: Multiple validation layers with graduated autonomy levels + command blocking
- **🧠 Knowledge Base**: Semantic code search with vector embeddings
- **⚡ Incremental Indexing**: 300x faster re-indexing by skipping unchanged files
- **🔌 Pluggable LLM**: Works with Claude, OpenAI, Ollama, or any OpenAI-compatible API
- **⚙️ Parallel Execution**: Runs independent module tasks concurrently
- **🎯 Dependency-Aware**: Respects module dependencies when ordering work

## 📦 Installation

```bash
# Base installation
pip install anton

# With Claude support (recommended)
pip install anton[claude]

# With OpenAI support
pip install anton[openai]

# With Ollama (local models)
pip install anton[ollama]

# All providers + knowledge base
pip install anton[all]
```

## 🚀 Quick Start

### 1. Initialize Your Repository

```bash
# Analyze your repository structure
anton init /path/to/your/repo

# This creates: .modular-agents/knowledge.json
```

### 2. Run a Task (with smart defaults)

```bash
# Set your API key
export ANTHROPIC_API_KEY="your-key"

# Run a task - automatically uses:
# - Checkpointing (can resume if interrupted)
# - Progress tracking (real-time monitoring)
# - Supervised autonomy (asks for risky changes)
# - Knowledge base (if indexed)
anton task . "Add user authentication"
```

### 3. Monitor Progress

```bash
# In another terminal, watch progress live
anton progress <task_id> --watch
```

### 4. Resume if Interrupted

```bash
# If task was interrupted (Ctrl+C, crash, network issue)
anton resume <task_id>
```

## 📚 Core Commands

### Repository Management

```bash
# Initialize repository analysis
anton init .

# Analyze structure (no LLM needed)
anton analyze .

# View as JSON
anton analyze . --json

# Export knowledge to markdown
anton export . --output docs/architecture.md

# Add custom metadata interactively
anton enrich .
```

### Task Execution

```bash
# Basic task (uses smart defaults)
anton task . "Add caching to UserRepository"

# Autonomous mode (auto-retry enabled by default)
anton task . "Refactor API layer" \
  --autonomous \
  --max-retries 5

# Disable auto-retry if needed
anton task . "Experimental change" \
  --no-auto-retry

# Interactive mode (asks before every action)
anton task . "Update database schema" \
  --autonomy-level interactive

# Resume from checkpoint
anton task . "Continue previous work" \
  --resume <task_id>

# Disable checkpointing (not recommended)
anton task . "Quick fix" --no-checkpoint
```

### Progress & Checkpoints

```bash
# View progress
anton progress <task_id>

# Live monitoring (updates every 2 seconds)
anton progress <task_id> --watch

# List all checkpoints
anton checkpoints list

# List for specific task
anton checkpoints list --task <task_id>

# Clean old checkpoints (7+ days)
anton checkpoints clean

# Clean older than 30 days
anton checkpoints clean --older-than 30
```

### Resume Task

```bash
# Resume from last checkpoint
anton resume <task_id>

# With specific provider/model
anton resume <task_id> --provider claude --model claude-sonnet-4
```

### Knowledge Base

```bash
# Setup knowledge base (one-time)
anton setup

# Index repository for semantic search
anton index .

# Re-index (automatic incremental - only changed files)
anton index .

# Force full re-index
anton index . --force

# Search code semantically
anton knowledge "user authentication logic"

# Search in specific repo/module
anton knowledge "caching" --repo /path --module api

# List all indexed projects
anton list-projects

# JSON output
anton list-projects --json
```

### Tool Calling (Enabled by Default)

Agents automatically use tools to gather context and validate changes:

```bash
# Tools enabled by default (no flag needed)
anton task . "Add caching to UserService"

# Agents will automatically:
# - Read existing code files
# - Search for similar patterns
# - Validate syntax before proposing
# - Run tests to verify changes

# Disable all tools
anton task . "task description" --no-tools

# Disable command execution only (read-only tools)
anton task . "task description" --no-command-execution
```

**Available Tools**:
- `read_file` - Read code files to understand implementation
- `grep_codebase` - Search for patterns and function definitions
- `search_knowledge_base` - Find similar code (requires indexed repo)
- `validate_syntax` - Check syntax before proposing changes
- `run_tests` - Execute tests to validate implementation
- `run_command` - Run build commands and linters
- `get_dependencies` - Extract module dependencies
- `get_git_history` - View file commit history

**Safety**: Dangerous commands (`rm -rf`, `format`, etc.) are automatically blocked.

See [docs/TOOL_CALLING.md](docs/TOOL_CALLING.md) for full documentation.

### Interactive Mode

```bash
# Start interactive REPL
anton run .

Commands in REPL:
  help              Show help
  modules           List all modules
  module <name>     Show module details
  <task>            Execute a task
  quit              Exit
```

## ⚙️ Configuration

### Smart Defaults (Enabled Automatically)

When you run `anton task`, these features are **enabled by default**:

✅ **Tool Calling**: Agents can read files, search code, run tests, validate syntax
✅ **Auto-Retry**: Automatically retry failed subtasks (up to 3 attempts)
✅ **Checkpointing**: Automatic save after each phase
✅ **Progress Tracking**: Real-time monitoring
✅ **Supervised Autonomy**: Auto-approve safe changes, ask for risky ones
✅ **Knowledge Base**: Semantic code search (if repository is indexed)
✅ **Safety Checks**: File limits, forbidden paths, sensitive patterns, command blocking
✅ **Validation**: Path checking, language enforcement, no placeholder code

You don't need to configure anything - just run tasks!

### Autonomy Levels

| Level | Flag | Behavior | Use Case |
|-------|------|----------|----------|
| **Interactive** | `--autonomy-level interactive` | Ask before every action | Critical changes, learning |
| **Supervised** | Default | Auto-approve safe, ask for risky | Most development tasks |
| **Autonomous** | `--autonomy-level autonomous` | Auto-approve with safety checks | Trusted refactorings |
| **Full** | `--autonomous` | Maximum automation | CI/CD pipelines |

### Environment Variables

```bash
# LLM Provider Keys
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
export LLM_API_KEY="generic-key"  # Fallback if provider-specific not set

# Knowledge Base (optional)
export ANTON_KB_PATH="~/.modular-agents/knowledge.db"  # Custom DB location
```

### Configuration Files

```
your-repo/
├── .modular-agents/
│   ├── knowledge.json         # Cached repository analysis
│   ├── custom.json            # User-provided enrichments
│   ├── checkpoints/           # Task checkpoints
│   ├── progress/              # Real-time progress tracking
│   ├── retries/               # Retry history and logs
│   └── summary_*.md           # Task execution summaries
```

## 🎯 Task Execution Flags

### Basic Options
- `--provider <name>` - LLM provider: claude, openai, ollama (default: claude)
- `--model <name>` - Model name (default: claude-sonnet-4-20250514)
- `--api-key <key>` - API key (or use environment variable)
- `--verbose, -v` - Show LLM interactions in real-time
- `--debug, -d` - Show full debug output with prompts
- `--dry-run` - Show execution plan without running

### Continuation Features (Enabled by Default)
- `--autonomous` - Full autonomous mode (no approval required)
- `--autonomy-level <level>` - Set level: interactive|supervised|autonomous|full
- `--auto-retry` - Enable automatic retry (default: disabled)
- `--max-retries <n>` - Maximum retry attempts (default: 3)
- `--resume <task_id>` - Resume from checkpoint
- `--no-checkpoint` - Disable automatic checkpointing

### Knowledge Base
- `--use-knowledge` - Enable knowledge base for code context (default: auto if indexed)
- `--kb-db <path>` - Knowledge base database path
- `--embedding-model <name>` - Embedding model for similarity search

## 🔧 LLM Provider Examples

### Claude (Anthropic) - Recommended

```bash
export ANTHROPIC_API_KEY="your-key"
anton task . "Add feature" --provider claude --model claude-sonnet-4
```

### OpenAI

```bash
export OPENAI_API_KEY="your-key"
anton task . "Add feature" --provider openai --model gpt-4o
```

### Ollama (Local)

```bash
# Start Ollama server first
ollama serve

# Run tasks
anton task . "Add feature" --provider ollama --model llama3.1
```

### OpenAI-Compatible (vLLM, LM Studio, etc.)

```bash

 # In your environment or .env file
 export CEREBRAS_API_KEY=csk-kn8yfhypj8kph2krjdydwtkf2wpx8h63hcrmwhpnxkh39dpk
 export CEREBRAS_BASE_URL=https://api.cerebras.ai/v1

  # Then use with Anton
  anton task . "add an api endpoint to query user by enrollment date" \
    --provider openai \
    --base-url $CEREBRAS_BASE_URL \
    --api-key $CEREBRAS_API_KEY \
    --model llama-3.3-70b

anton task . "add an api endpoint to query user by enrollment date" \
  --provider openai \
  --model local \
  --base-url http://localhost:8000/v1 \
  --api-key "not-needed"
```

## 📖 Usage Examples

### Long-Running Autonomous Task

```bash
# Start autonomous refactoring with auto-retry
anton task . "Refactor authentication system" \
  --autonomous \
  --auto-retry \
  --max-retries 5

# If interrupted (Ctrl+C, crash), resume later
anton resume <task_id>

# Monitor in another terminal
anton progress <task_id> --watch
```

### Interactive Critical Changes

```bash
# Ask before every change
anton task . "Migrate database schema" \
  --autonomy-level interactive

# For each subtask, shows:
# - Proposed changes
# - Files to modify
# - Safety warnings
# - Waits for yes/no approval
```

### CI/CD Pipeline Integration

```bash
# Full autonomy for automated workflows
#!/bin/bash
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}"

anton task /workspace "Update dependencies and fix breaking changes" \
  --autonomous \
  --auto-retry \
  --max-retries 5 \
  --verbose

# Automatically:
# - Runs without approval
# - Retries failures with exponential backoff
# - Creates checkpoints (can resume if fails)
# - Logs all interactions
```

### Knowledge-Enhanced Development

```bash
# Index your repository first
anton init .
anton index .

# Run task with knowledge base (finds existing patterns)
anton task . "Add request validation to API endpoints" \
  --use-knowledge \
  --autonomous

# Agent will:
# - Search for existing validation patterns
# - Find similar endpoint implementations
# - Learn from existing code style
# - Apply consistent patterns
```

### Multi-Terminal Workflow

```bash
# Terminal 1: Run long task
anton task . "Large refactoring" --autonomous --auto-retry

# Terminal 2: Monitor progress
anton progress <task_id> --watch

# Terminal 3: Check checkpoints
anton checkpoints list --task <task_id>

# Terminal 4: Search knowledge base
anton knowledge "error handling patterns"
```

## 🛡️ Safety & Control

### Safety Checks (Always Active)

Regardless of autonomy level, Anton enforces:

- **File Limits**: Max 10 files per subtask, 1000 lines per file
- **Forbidden Paths**: `.git/`, `.env`, `secrets/`, `credentials/`, `.ssh/`
- **Sensitive Patterns**: `API_KEY`, `SECRET`, `PASSWORD`, `TOKEN` detection
- **Critical Violations**: Blocks execution immediately

### Approval Requirements (Supervised Mode)

Anton asks for approval when:

- ❌ Deleting files
- 📏 Large changes (> 500 lines)
- ⚙️ Config file modifications (`.yaml`, `.json`, `.toml`)
- 📦 Dependency updates (`package.json`, `requirements.txt`, `build.sbt`)
- 🔀 Multi-module changes (> 3 modules affected)

### What Gets Checkpointed

Every checkpoint saves:
- ✅ Full execution plan
- ✅ Completed subtask results
- ✅ Pending subtasks
- ✅ Agent conversation history (for continuation)
- ✅ Metadata (module mappings, descriptions)

## 📊 Performance & Cost

### Continuation System Overhead
- **Checkpointing**: ~50-100ms per save, ~10-50 KB per checkpoint
- **Progress Tracking**: ~10-20ms per update, ~5 KB per task
- **Retry Logic**: 0 overhead for successful subtasks
- **Overall**: < 5% overhead for typical tasks

### Knowledge Base
- **Incremental Indexing**: 300x faster for unchanged files
- **Cost Savings**: 95%+ reduction in LLM API calls for re-indexing
- **Disk Usage**: ~60-120 MB per 1000 files

### Example Savings

**Scenario**: 100 files, 5 modified, re-indexing
- **Without Incremental**: 100 LLM API calls, ~$1.00
- **With Incremental**: 5 LLM API calls, ~$0.05
- **Savings**: 95% cost reduction, 300x faster

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Task                            │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    OrchestratorAgent                        │
│  • Decomposes tasks into module-specific subtasks           │
│  • Creates execution plan respecting dependencies           │
│  • Manages checkpoints & retry logic                        │
│  • Tracks progress in real-time                             │
│  • Queries knowledge base for context                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ ModuleAgent │   │ ModuleAgent │   │ ModuleAgent │
│   (api)     │   │  (domain)   │   │(persistence)│
│             │   │             │   │             │
│ • Specialized│   │ • Preserves │   │ • Respects  │
│ • Saves state│   │   patterns  │   │   boundaries│
└─────────────┘   └─────────────┘   └─────────────┘
```

## 📁 File Structure

After using Anton:

```
your-repo/
├── .modular-agents/
│   ├── knowledge.json              # Repository structure analysis
│   ├── custom.json                 # User enrichments (optional)
│   ├── checkpoints/
│   │   └── <task_id>/
│   │       ├── checkpoint_phase0_*.json
│   │       ├── checkpoint_phase1_*.json
│   │       └── metadata.json
│   ├── progress/
│   │   └── <task_id>.json         # Real-time progress
│   ├── retries/
│   │   ├── <task_id>_retries.jsonl
│   │   └── <task_id>_<subtask>_history.json
│   ├── traces/                     # LLM interactions (if --verbose)
│   │   ├── orchestrator_*.jsonl
│   │   ├── module_*_*.jsonl
│   │   └── trace_summary.json
│   └── summary_<task_id>.md       # Task summaries
```

## 🔍 Troubleshooting

### Task Won't Resume

```bash
# Check if checkpoints exist
anton checkpoints list --task <task_id>

# Verify checkpoint directory
ls .modular-agents/checkpoints/<task_id>/
```

### Too Many Retries

```bash
# Check error logs
cat .modular-agents/retries/<task_id>_retries.jsonl

# Run with debug mode
anton task . "description" --debug
```

### Knowledge Base Not Working

```bash
# Verify setup
anton list-projects

# Re-index repository
anton index . --force

# Test search
anton knowledge "test query"
```

### Slow Indexing

```bash
# Use simple parser (no LLM, faster)
anton index . --simple-parser

# Skip embeddings initially
anton index . --no-embeddings
```

## 📚 Documentation

- [Agent Continuation & Autonomy](docs/CONTINUATION.md) - Checkpointing, retry, progress tracking
- [Knowledge Base Guide](docs/KNOWLEDGE_BASE.md) - Semantic search, incremental indexing
- [Development Guide](docs/DEVELOPMENT.md) - Contributing, architecture, testing

## 🎓 Supported Project Types

| Type | Build File | Status |
|------|------------|--------|
| SBT (Scala) | `build.sbt` | ✅ Full support |
| Maven (Java) | `pom.xml` | ✅ Full support |
| npm (JavaScript) | `package.json` | ✅ Full support |
| Poetry (Python) | `pyproject.toml` | ✅ Full support |
| Cargo (Rust) | `Cargo.toml` | ✅ Full support |
| Go | `go.mod` | ✅ Full support |
| Gradle | `build.gradle` | 🔜 Coming soon |

Generic analyzer can handle any project structure using LLM-based analysis.

## ❓ FAQ

**Q: Do I need to enable checkpointing?**
A: No, it's enabled by default. Just run `anton task` and you can resume if interrupted.

**Q: How do I make tasks fully autonomous?**
A: Use `--autonomous` flag: `anton task . "description" --autonomous`

**Q: Can I use local LLMs?**
A: Yes! Use Ollama or any OpenAI-compatible server with `--provider ollama` or `--base-url`.

**Q: How much does the knowledge base cost?**
A: First indexing costs ~$0.01-0.10 per 100 files. Re-indexing is 95% cheaper with incremental updates.

**Q: Is my code sent to external servers?**
A: Only if using cloud LLM providers (Claude, OpenAI). Use Ollama for fully local processing.

**Q: Can I use this in CI/CD?**
A: Yes! Use `--autonomous --auto-retry` for automated pipelines.

**Q: What if task fails with errors?**
A: Use `--auto-retry` to automatically retry with exponential backoff. Check logs in `.modular-agents/retries/`.

## 🤝 Contributing

```bash
# Clone the repository
git clone https://github.com/yourname/anton
cd anton

# Install in development mode
pip install -e ".[all,dev]"

# Run tests
pytest

# Lint code
ruff check src/
```

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for detailed contribution guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built with:
- [Anthropic Claude](https://www.anthropic.com) - Primary LLM provider
- [LangChain](https://langchain.com) - LLM orchestration
- [Rich](https://rich.readthedocs.io/) - Terminal UI
- [Click](https://click.palletsprojects.com/) - CLI framework
- [Sentence Transformers](https://www.sbert.net/) - Embeddings
- [hnswlib](https://github.com/nmslib/hnswlib) - Vector search

---

**Made with ❤️ for developers who want autonomous, fault-tolerant code development**
