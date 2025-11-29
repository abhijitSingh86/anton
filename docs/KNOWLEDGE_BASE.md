# Knowledge Base

Anton includes a powerful knowledge base system that enables semantic code search and intelligent context retrieval using vector embeddings.

## Features

- **Semantic Code Search**: Find code by meaning, not just keywords
- **Vector Embeddings**: HNSW-based fast similarity search
- **Incremental Indexing**: Only re-index changed files (up to 300x faster)
- **Multi-Repository**: Index and search across multiple projects
- **LLM-Assisted Parsing**: Intelligent code chunking with purpose extraction
- **Automatic Context**: Agents can query knowledge base during task execution

## Quick Start

### Setup Knowledge Base

```bash
# Initialize the global knowledge base
anton setup

# This creates: ~/.modular-agents/knowledge.db
```

### Index a Repository

```bash
# Initialize repository analysis first
anton init /path/to/your/project

# Index the repository
anton index /path/to/your/project

# Subsequent re-indexing is automatic (only changed files)
anton index /path/to/your/project  # Fast incremental update
```

### Search Code

```bash
# Search for code by semantic meaning
anton knowledge "user authentication logic"

# Filter by repository
anton knowledge "caching implementation" --repo /path/to/project

# Filter by module
anton knowledge "error handling" --module api

# Limit results
anton knowledge "database queries" --limit 20
```

### List Indexed Projects

```bash
# View all indexed repositories
anton list-projects

# JSON output for scripting
anton list-projects --json
```

## Incremental Indexing

### How It Works

Anton tracks file modification times and automatically skips unchanged files:

1. **First indexing**: All files are indexed
2. **Subsequent indexing**: Only changed/new files are re-indexed
3. **Deleted files**: Automatically removed from index
4. **Force re-index**: Use `--force` flag to override

### Performance Benefits

| Scenario | Files | Without Incremental | With Incremental | Speedup |
|----------|-------|---------------------|------------------|---------|
| No changes | 100 | 5 minutes | < 1 second | 300x |
| 5% changed | 100 | 5 minutes | 15 seconds | 20x |
| 20% changed | 100 | 5 minutes | 1 minute | 5x |

### Cost Savings

**Example**: 100 files, 5 modified
- **Old**: 100 LLM API calls
- **New**: 5 LLM API calls
- **Savings**: 95% reduction in costs

### Usage

```bash
# Incremental update (default - skips unchanged files)
anton index .

# Force full re-index
anton index . --force

# View what was skipped
anton index .
# Output:
# Skipped 95 unchanged files
# Files to process: 5
# Files indexed: 5
# Chunks created: 14
```

## Configuration

### Embedding Models

Default: `sentence-transformers/all-MiniLM-L6-v2`

```bash
# Use different embedding model
anton setup --model sentence-transformers/all-mpnet-base-v2

# Use GPU
anton setup --device cuda

# Use Apple Silicon
anton setup --device mps
```

### LLM for Parsing

```bash
# Use Claude (default)
anton index . --provider claude

# Use OpenAI
anton index . --provider openai

# Use local Ollama
anton index . --provider ollama --base-url http://localhost:11434

# Use simple regex parser (no LLM, faster but less accurate)
anton index . --simple-parser
```

### Database Location

```bash
# Default location
~/.modular-agents/knowledge.db

# Custom location
anton setup --db /custom/path/knowledge.db
anton index . --db /custom/path/knowledge.db
anton knowledge "query" --db /custom/path/knowledge.db
```

## Search Features

### Semantic Search

Search by meaning, not exact text:

```bash
# Finds: login(), authenticate(), verifyUser(), etc.
anton knowledge "user authentication"

# Finds: cache.get(), memoize(), stored results, etc.
anton knowledge "caching mechanisms"

# Finds: try/catch, error handlers, validation, etc.
anton knowledge "error handling patterns"
```

### Filters

```bash
# Search in specific repository
anton knowledge "query" --repo /path/to/project

# Search in specific module
anton knowledge "query" --module api

# Limit number of results
anton knowledge "query" --limit 5
```

### Output Format

Results show:
- **Code chunk**: Syntax-highlighted source code
- **Score**: Similarity score (0-1)
- **Metadata**: Module, file path, line numbers
- **Summary**: LLM-generated purpose (if available)
- **Chunk type**: Function, class, method, etc.

## Agent Integration

### Enable Knowledge Base for Tasks

```bash
# Run task with knowledge base access
anton task . "Add caching" --use-knowledge

# Specify custom database
anton task . "Add caching" --use-knowledge --kb-db /custom/path/knowledge.db
```

### How Agents Use Knowledge

When knowledge base is enabled:
1. **Context Retrieval**: Agents search for relevant code before implementing
2. **Pattern Learning**: Find existing patterns to match
3. **Dependency Discovery**: Understand how code is used elsewhere
4. **Example-Based**: Learn from existing implementations

## Command Reference

### Setup

```bash
anton setup [options]

Options:
  --db PATH              Database path (default: ~/.modular-agents/knowledge.db)
  --model MODEL          Embedding model (default: all-MiniLM-L6-v2)
  --device DEVICE        Device: cpu|cuda|mps (default: cpu)
```

### Index

```bash
anton index <path> [options]

Options:
  --db PATH              Database path
  --provider PROVIDER    LLM provider: claude|openai|ollama (default: claude)
  --model MODEL          LLM model name
  --force, -f            Force full re-index (skip change detection)
  --simple-parser        Use regex parser instead of LLM (faster, no API cost)
  --no-embeddings        Skip generating embeddings (faster indexing)
  --debug                Show detailed errors
```

### Search

```bash
anton knowledge "<query>" [options]

Options:
  --db PATH              Database path
  --limit N              Number of results (default: 10)
  --repo PATH            Filter by repository path
  --module NAME          Filter by module name
  --embedding-model      Embedding model (default: all-MiniLM-L6-v2)
  --device DEVICE        Device for embeddings (default: cpu)
```

### List Projects

```bash
anton list-projects [options]

Options:
  --db PATH              Database path
  --json                 Output as JSON
```

## Database Schema

### Tables

**indexed_repos**
- Repository metadata
- Project type, language
- Index timestamps
- File and chunk counts

**indexed_files**
- File-level tracking
- Modification times (mtime)
- Module and language
- Chunk counts

**code_chunks**
- Parsed code segments
- Content, purpose, summary
- Module and file associations
- Chunk types (function, class, etc.)

**embeddings**
- Vector embeddings
- HNSW index for fast search
- Linked to code chunks

**relations**
- Code dependencies
- Cross-references
- Relationships between chunks

**task_learnings**
- Task execution history
- Agent learnings
- Success/failure patterns

## Examples

### Index Multiple Projects

```bash
# Index your main project
anton init ~/projects/my-app
anton index ~/projects/my-app

# Index a library you're using
anton init ~/projects/shared-lib
anton index ~/projects/shared-lib

# Index another service
anton init ~/projects/api-service
anton index ~/projects/api-service

# View all indexed projects
anton list-projects
```

### Search Across Projects

```bash
# Search all indexed projects
anton knowledge "rate limiting implementation"

# Search specific project
anton knowledge "rate limiting" --repo ~/projects/api-service
```

### Update After Code Changes

```bash
# Make code changes...
# Edit files, add new files, delete files

# Quick incremental update
anton index .

# Output:
# Updating Repository: /home/user/project
# Skipped 95 unchanged files
# Files to process: 5
# Deleted files: 2
#
# Indexing Complete!
#   Files indexed: 5
#   Chunks created: 14
#   Skipped unchanged: 95
#   Deleted files: 2
```

### Use in Task Execution

```bash
# Enable knowledge base for better context
anton task . "Add request validation to API endpoints" \
  --use-knowledge \
  --autonomous

# Agent will:
# 1. Search for existing validation patterns
# 2. Find similar API endpoint implementations
# 3. Learn from existing code style
# 4. Apply consistent patterns
```

## Storage Requirements

### Disk Space

**Per 1000 files**:
- Code chunks: ~5-10 MB
- Embeddings: ~50-100 MB
- Metadata: ~1-2 MB
- **Total**: ~60-120 MB

### Database Size Examples

| Project Size | Files | Database Size |
|--------------|-------|---------------|
| Small | 100 | 6-12 MB |
| Medium | 500 | 30-60 MB |
| Large | 2000 | 120-240 MB |
| Very Large | 10000 | 600 MB - 1.2 GB |

## Performance

### Indexing Speed

**With LLM Parser**:
- ~5-10 files/minute (depends on LLM API)
- Network-bound (API latency)
- Cost: ~$0.001-0.01 per file

**With Simple Parser**:
- ~100-500 files/minute
- CPU-bound (regex parsing)
- Cost: Free (no API calls)

### Search Speed

- Vector search: < 100ms for typical queries
- HNSW index provides logarithmic scaling
- Filters (repo, module) applied efficiently

## Best Practices

1. **Index after setup**: Run `anton index .` after cloning new project
2. **Re-index periodically**: After major changes or branch switches
3. **Use incremental**: Let automatic change detection work (default)
4. **Force when needed**: Use `--force` after switching branches
5. **Clean periodically**: Remove old project indexes you no longer need
6. **Enable for tasks**: Use `--use-knowledge` for better agent context
7. **Filter searches**: Use `--repo` and `--module` to narrow results
8. **Choose parser**: LLM parser for accuracy, simple parser for speed

## Troubleshooting

### Slow Indexing

```bash
# Use simple parser (no LLM calls)
anton index . --simple-parser

# Skip embeddings during indexing
anton index . --no-embeddings

# Index later
anton index . --simple-parser
# Then generate embeddings
anton index . --force
```

### Out of Date Results

```bash
# Force full re-index
anton index . --force

# Check when last indexed
anton list-projects
```

### Large Database Size

```bash
# Remove old projects
rm -rf ~/.modular-agents/knowledge.db

# Re-initialize and index only current projects
anton setup
anton index ~/current/project
```

### Search Returns No Results

1. Verify project is indexed: `anton list-projects`
2. Re-index if needed: `anton index .`
3. Try broader search terms
4. Check filters (remove `--repo` or `--module`)

### Incremental Indexing Not Working

```bash
# Verify file tracking
ls .modular-agents/

# Force re-index
anton index . --force

# Check for errors
anton index . --debug
```
