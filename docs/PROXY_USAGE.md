# Proxy-Based Tracing

Anton supports **transparent LLM tracing** through a proxy server. This approach keeps the Anton source code clean while providing comprehensive observability.

The proxy sits between Anton and your LLM backend (official APIs or local LLMs), logging all calls to a database without requiring any code changes.

## Quick Start

### 1. Install Proxy Dependencies

```bash
pip install 'anton[proxy]'
```

### 2. Start the Proxy Server

**For local LLM (OpenAI-compatible):**
```bash
anton proxy start --openai-backend http://localhost:8000/v1
# Proxy running on http://localhost:8001
# OpenAI backend: http://localhost:8000/v1
```

**For official APIs (default):**
```bash
anton proxy start
# Proxy running on http://localhost:8001
# Forwards to official OpenAI/Claude APIs
```

### 3. Start the Dashboard (optional)

In another terminal:

```bash
anton proxy dashboard
# Dashboard running on http://localhost:3000
```

### 4. Configure Anton to Use Proxy

**Only set once!** Then use Anton normally.

```bash
# For OpenAI or local LLM
export OPENAI_BASE_URL=http://localhost:8001

# For Claude
export ANTHROPIC_BASE_URL=http://localhost:8001
```

### 5. Use Anton Normally - No --base-url Flag Needed!

```bash
# Notice: NO --base-url flag!
anton task . "Your task" --provider openai --model "your-model" --api-key "your-key"
```

**All LLM calls are now traced!** View them in the dashboard at http://localhost:3000

## Architecture

```
┌──────────────────────┐
│   Anton Task         │
│   (Clean Source)     │
│   export OPENAI_     │
│   BASE_URL=:8001     │
└──────────┬───────────┘
           │ LLM API call
           ↓
┌──────────────────────┐
│   Proxy Server       │
│   (localhost:8001)   │
│   ├─ Logs call       │
│   └─ Forwards →      │
└──────────┬───────────┘
           │ Forward to backend
           ↓
┌──────────────────────┐
│   Backend LLM        │
│   (configured at     │
│    proxy start)      │
│   • Local: :8000/v1  │
│   • OpenAI API       │
│   • Claude API       │
└──────────────────────┘

           ↓ Logs stored
┌──────────────────────┐
│   SQLite Database    │
│   (~/.modular-agents │
│    /proxy_traces.db) │
└──────────┬───────────┘
           │ Queries
           ↓
┌──────────────────────┐
│   Dashboard          │
│   (localhost:3000)   │
└──────────────────────┘
```

## Features

### Transparent Logging
- Zero code changes required in Anton
- Captures all LLM requests and responses
- Records timing, tokens, and costs
- Supports multiple LLM providers (Claude, OpenAI)

### Real-Time Dashboard
- View traces as they happen
- Browse LLM calls with full details
- See request/response data
- Track token usage and costs

### Database Storage
- All traces stored in SQLite
- Persistent across sessions
- Queryable for analysis
- Located at `~/.modular-agents/proxy_traces.db`

## CLI Commands

### Start Proxy

```bash
anton proxy start [OPTIONS]

Options:
  --host TEXT              Host to bind to (default: localhost)
  --port INTEGER           Port to bind to (default: 8001)
  --openai-backend TEXT    OpenAI backend URL (default: official API)
  --claude-backend TEXT    Claude backend URL (default: official API)

Examples:
  # With local LLM
  anton proxy start --openai-backend http://localhost:8000/v1

  # With official APIs
  anton proxy start

  # Custom port
  anton proxy start --port 9001 --openai-backend http://localhost:8000/v1
```

### Start Dashboard

```bash
anton proxy dashboard [OPTIONS]

Options:
  --host TEXT     Host to bind to (default: localhost)
  --port INTEGER  Port to bind to (default: 3000)
```

## Configuration

### Environment Variables

**Set these ONCE, then use Anton normally:**

```bash
# For OpenAI/Local LLM - points Anton to the proxy
export OPENAI_BASE_URL=http://localhost:8001
export OPENAI_API_KEY=your-key  # Your actual API key (or "not-needed" for local LLMs)

# For Claude - points Anton to the proxy
export ANTHROPIC_BASE_URL=http://localhost:8001
export ANTHROPIC_API_KEY=your-key  # Your actual API key
```

The proxy receives requests from Anton and forwards them to the backend configured at proxy start.

### Complete Example: Local LLM

```bash
# 1. Start your local LLM server (e.g., Ollama, LMStudio)
# Assume it's running on http://localhost:8000/v1

# 2. Start proxy pointing to your local LLM
anton proxy start --openai-backend http://localhost:8000/v1

# 3. Start dashboard (optional)
anton proxy dashboard

# 4. Configure Anton (one time setup)
export OPENAI_BASE_URL=http://localhost:8001
export OPENAI_API_KEY="not-needed"

# 5. Use Anton normally - NO --base-url flag!
anton task . "Create a hello world function" \
  --provider openai \
  --model "models/qwen2.5-coder-14b-instruct"

# All calls are now logged and visible in dashboard!
```

### Custom Ports

```bash
# Proxy on port 9001 with local LLM backend
anton proxy start --port 9001 --openai-backend http://localhost:8000/v1

# Dashboard on port 4000
anton proxy dashboard --port 4000

# Update environment to match proxy port
export OPENAI_BASE_URL=http://localhost:9001
```

## Benefits

✅ **Clean Source Code**
- No tracing code in Anton
- No performance overhead when proxy disabled
- Easy to maintain

✅ **Transparent Operation**
- Works with existing code
- No modifications needed
- Captures all LLM calls automatically

✅ **Flexible Usage**
- Enable/disable via environment variables
- Works with any LLM provider
- Centralized logging

✅ **Structured Storage**
- SQLite database for queries
- Efficient storage
- Easy to export/analyze

## Troubleshooting

### Proxy not intercepting calls

Check that environment variables are set:

```bash
echo $ANTHROPIC_BASE_URL
echo $OPENAI_BASE_URL
```

### Dashboard shows no traces

1. Ensure proxy is running
2. Check database exists: `ls ~/.modular-agents/proxy_traces.db`
3. Run a task to generate traces

### Connection errors

Check ports are available:

```bash
lsof -i :8001  # Proxy port
lsof -i :3000  # Dashboard port
```

## Database Schema

The proxy uses the following tables:

- **traces**: One row per task execution
- **llm_calls**: All LLM API calls with full details
- **call_graph**: Relationships between calls (for future visualization)

Query directly:

```bash
sqlite3 ~/.modular-agents/proxy_traces.db "SELECT * FROM traces LIMIT 10"
```

## Disable Tracing

To disable tracing, simply unset the environment variable(s) you set:

```bash
unset OPENAI_BASE_URL      # If you set this
unset ANTHROPIC_BASE_URL   # If you set this
```

Anton will make direct API calls without logging.

## Note on Port 8000

If you're running a local LLM on port 8000 (common for Ollama, LMStudio, etc.), the proxy defaults to port 8001 to avoid conflicts. You can always specify a custom port with `--port`.
