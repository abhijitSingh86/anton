# Agent Continuation & Autonomy

Anton includes a comprehensive continuation system that enables long-running, fault-tolerant task execution with automatic recovery.

## Features

- **Automatic Checkpointing**: Save progress after each phase
- **Resume Capability**: Continue from any checkpoint after interruption
- **Automatic Retry**: Exponential backoff for transient failures
- **Real-Time Progress**: Track task completion and estimate time remaining
- **Graduated Autonomy**: 4 levels from interactive to fully autonomous
- **Safety Checks**: Multiple validation layers to prevent dangerous operations

## Quick Start

### Basic Usage (All Enabled by Default)

```bash
# Run a task - these features are AUTOMATIC:
anton task . "Add user authentication"

# Task automatically includes:
# ✅ Checkpointing after each phase
# ✅ Auto-retry (up to 3 attempts)
# ✅ Progress tracking in real-time
# ✅ Supervised autonomy (safe defaults)
# ✅ Tool calling (read files, run tests, etc.)
```

### Autonomous Mode with Custom Retry Limit

```bash
# For long-running tasks that can run unsupervised
anton task . "Refactor API layer" \
  --autonomous \
  --max-retries 5

# Disable auto-retry if needed (not recommended)
anton task . "Experimental change" \
  --no-auto-retry
```

### Resume After Interruption

```bash
# If task was interrupted (Ctrl+C, crash, network issue)
# List available checkpoints
anton checkpoints list

# Resume from last checkpoint
anton resume <task_id>
```

### Monitor Progress

```bash
# In terminal 1: Run task
anton task . "Large refactoring" --autonomous

# In terminal 2: Watch progress live
anton progress <task_id> --watch
```

## Autonomy Levels

### Interactive (`--autonomy-level interactive`)
- **Control**: Maximum - asks before every action
- **Use case**: Critical changes, learning the system
- **Behavior**: Shows proposed changes and waits for approval (yes/no)

### Supervised (Default)
- **Control**: High - auto-approves safe actions, asks for risky ones
- **Use case**: Most development tasks
- **Behavior**: Automatically approves:
  - Small file changes (< 500 lines)
  - Single module updates
  - Test additions
  - **Asks approval for**:
  - File deletions
  - Large changes (> 500 lines)
  - Config file modifications
  - Dependency updates
  - Multi-module changes

### Autonomous (`--autonomy-level autonomous`)
- **Control**: Low - auto-approves with safety checks
- **Use case**: Trusted refactorings, well-defined tasks
- **Behavior**: Automatically approves all changes that pass safety checks

### Full (`--autonomous`)
- **Control**: Minimal - maximum automation
- **Use case**: CI/CD pipelines, overnight runs
- **Behavior**: No approval required, only critical safety violations block execution

## Automatic Retry

Enable automatic retry for transient failures:

```bash
anton task . "Deploy feature" --auto-retry --max-retries 3
```

**Retry Strategy**:
- Exponential backoff: 1s, 2s, 4s, 8s, 16s...
- Jitter added to prevent thundering herd
- Smart error classification (retryable vs. permanent)
- Full retry history logged

**Retryable Errors**:
- Network timeouts
- Rate limiting (429)
- Temporary service unavailability (503)
- LLM provider errors

**Non-Retryable Errors**:
- Authentication failures (401, 403)
- Invalid requests (400)
- Not found (404)
- JSON parsing errors

## Checkpointing

### Automatic Checkpoints

Enabled by default. Checkpoints are saved:
- After each execution phase completes
- Before starting next phase
- Includes full agent state and conversation history

```bash
# Disable if needed (not recommended)
anton task . "Quick fix" --no-checkpoint
```

### Checkpoint Management

```bash
# List all tasks with checkpoints
anton checkpoints list

# List checkpoints for specific task
anton checkpoints list --task <task_id>

# Clean old checkpoints (default: 7 days)
anton checkpoints clean

# Clean checkpoints older than 30 days
anton checkpoints clean --older-than 30
```

### Checkpoint Storage

```
.modular-agents/
└── checkpoints/
    └── <task_id>/
        ├── checkpoint_phase0_<id>.json
        ├── checkpoint_phase1_<id>.json
        └── metadata.json
```

## Progress Tracking

### View Progress

```bash
# Single snapshot
anton progress <task_id>

# Live updates (refreshes every 2 seconds)
anton progress <task_id> --watch
```

### Progress Information

Shows:
- Task description and status
- Phase completion (e.g., 2/3 phases)
- Subtask completion (e.g., 8/12 subtasks, 66.7%)
- Completed/Failed/Blocked counts
- Elapsed time
- Estimated remaining time
- Estimated completion time
- Total retry attempts
- Latest checkpoint path

### Progress Storage

```
.modular-agents/
└── progress/
    └── <task_id>.json
```

## Safety Checks

Safety checks are **always active** regardless of autonomy level:

### File Limits
- Max 10 files per subtask
- Max 1000 lines per file
- Max 5000 total lines per subtask

### Forbidden Paths
- `.git/` - Version control internals
- `.env` - Environment variables
- `secrets/` - Secret storage
- `credentials/` - Credentials
- `private/` - Private data
- `.ssh/` - SSH keys

### Sensitive Patterns
Files containing these patterns require approval:
- `API_KEY`, `SECRET`, `PASSWORD`, `TOKEN`
- `PRIVATE_KEY`, `CREDENTIALS`
- `aws_secret_access_key`, `GITHUB_TOKEN`

### Critical Violations

Execution is **blocked** if:
- Attempting to modify forbidden paths
- Sensitive patterns detected in code
- Safety limits exceeded

## Command Reference

### Task Execution

```bash
anton task <path> "<description>" [options]

Options:
  --autonomous              Enable full autonomous mode
  --autonomy-level LEVEL    Set level: interactive|supervised|autonomous|full
  --auto-retry              Enable automatic retry
  --max-retries N          Maximum retry attempts (default: 3)
  --resume TASK_ID         Resume from checkpoint
  --no-checkpoint          Disable automatic checkpointing
  --verbose, -v            Show LLM interactions
  --debug, -d              Show full debug output
```

### Resume

```bash
anton resume <task_id> [options]

Options:
  --path PATH              Repository path (default: .)
  --provider PROVIDER      LLM provider (default: claude)
  --model MODEL            Model name
  --verbose, -v            Show LLM interactions
```

### Checkpoints

```bash
# List all checkpoints
anton checkpoints list

# List for specific task
anton checkpoints list --task <task_id>

# Clean old checkpoints
anton checkpoints clean [--older-than DAYS] [--task TASK_ID]
```

### Progress

```bash
# View progress
anton progress <task_id>

# Live monitoring
anton progress <task_id> --watch
```

## Recent Enhancements (2025-11-29)

### Agent Scratchpad System

Each agent now has a **scratchpad** - working memory that tracks discoveries, planning, and validation throughout task execution.

**Features**:
- **Discovery Tracking**: Records files found, patterns discovered, and knowledge base results
- **Planning Phase**: Tracks files to create/modify, dependencies needed, and implementation approach
- **Validation Gates**: Four automatic checkpoints:
  - Path validation (ensures files stay within module boundaries)
  - Duplicate detection (prevents creating the same file twice)
  - Placeholder check (blocks code with TODOs or "not implemented" stubs)
  - Language validation (ensures correct file extensions for language)
- **Tool Usage Tracking**: Records which tools were used and what they discovered
- **Automatic Retry with Feedback**: If validation fails, agent gets scratchpad summary for next attempt

**Benefits**:
- Better debugging - see exactly what agent discovered and why it made decisions
- Fewer errors - validation gates catch common mistakes early
- Smarter retries - agent learns from previous attempt via scratchpad feedback

### Language Detection for Empty Repositories

When starting a task in an empty repository, Anton now automatically detects the programming language from your task description.

**Example**:
```bash
# Empty directory - no files to analyze
mkdir my-new-service && cd my-new-service

# Anton detects "Scala" from task description
anton task . "Create a Scala User case class with id, name, email fields"

# Output shows:
# ✓ Detected language from task: scala
# Creates: User.scala (not User.unk!)
```

**Supported Languages**: Scala, Python, Java, TypeScript, JavaScript, Rust, Go, C++, C#, Ruby

**Fallback**: In non-autonomous mode, if language can't be detected, Anton will prompt you to select from supported languages.

### Knowledge Base Enhancements

The knowledge base system has been improved to enable **cross-repository learning**.

**What Changed**:
- Knowledge base is now **always enabled** if it exists (previously only for indexed repos)
- New repositories can learn from previously indexed repositories
- Fixed embeddings generation (2,915 embeddings for semantic search)
- Corrected database location: `~/.modular-agents/knowledge.db`

**Benefits**:
```bash
# Index your main project
anton index /path/to/main-project

# Work on NEW project - can learn from main project!
cd /path/to/new-project
anton task . "Create a repository like in main-project"
# Agent will use search_knowledge_base to find similar patterns
```

**Example Use Case**:
1. Index a mature Scala project with good patterns
2. Start a new Scala microservice
3. Agents automatically reference the indexed project for:
   - Repository trait patterns
   - JSON codec patterns
   - ZIO HTTP endpoint structures
   - Test patterns

See `docs/KNOWLEDGE_BASE.md` for indexing commands.

## Examples

### Long-Running Refactoring

```bash
# Start autonomous refactoring with retry
anton task . "Refactor entire authentication system" \
  --autonomous \
  --auto-retry \
  --max-retries 5

# If interrupted, resume later
anton resume <task_id>
```

### Interactive Critical Changes

```bash
# Interactive mode for database schema changes
anton task . "Update user table schema" \
  --autonomy-level interactive
```

### Overnight CI/CD Task

```bash
# Full autonomy for automated pipeline
anton task . "Update all dependencies and fix breaking changes" \
  --autonomous \
  --auto-retry
```

### Monitor Long Task

```bash
# Terminal 1: Start task
anton task . "Migrate to new API version" --autonomous &

# Terminal 2: Monitor
anton progress <task_id> --watch
```

## File Structure

After using continuation features:

```
your-repo/
├── .modular-agents/
│   ├── checkpoints/
│   │   └── <task_id>/
│   │       ├── checkpoint_phase0_*.json
│   │       ├── checkpoint_phase1_*.json
│   │       └── metadata.json
│   ├── progress/
│   │   └── <task_id>.json
│   ├── retries/
│   │   ├── <task_id>_retries.jsonl
│   │   └── <task_id>_<subtask_id>_history.json
│   └── summary_<task_id>.md
```

## Best Practices

1. **Use supervised mode** (default) for most tasks
2. **Enable auto-retry** for network-dependent operations
3. **Monitor progress** for long-running tasks
4. **Clean checkpoints** periodically to save disk space
5. **Use interactive mode** when learning or for critical changes
6. **Use autonomous mode** only for well-defined, trusted tasks
7. **Keep checkpointing enabled** unless disk space is extremely limited

## Troubleshooting

### Task Won't Resume

```bash
# Check if checkpoints exist
anton checkpoints list --task <task_id>

# Verify checkpoint directory
ls .modular-agents/checkpoints/<task_id>/
```

### Progress Not Updating

Progress files are deleted when tasks complete. Check:
```bash
ls .modular-agents/progress/
```

### Too Many Retries

If a subtask keeps failing:
1. Check error logs in `.modular-agents/retries/`
2. Review safety violations
3. Consider reducing task scope
4. Try with `--debug` flag for detailed output

### Disk Space

Clean old data:
```bash
# Remove checkpoints older than 7 days
anton checkpoints clean

# Remove all temporary files
rm -rf .modular-agents/checkpoints/*
rm -rf .modular-agents/progress/*
rm -rf .modular-agents/retries/*
```
