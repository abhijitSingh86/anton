# Observability Dashboard

The Anton Dashboard provides real-time observability into your multi-agent task executions with timeline visualization, cost tracking, and performance metrics.

## Features

- **Timeline Visualization**: Gantt-style view of LLM calls and tool executions
- **Hierarchical Traces**: Parent-child relationships between spans
- **Cost Tracking**: Per-call and aggregate cost estimation
- **Token Usage**: Input/output token counts for all LLM interactions
- **Error Tracking**: Failed operations with error messages
- **Performance Metrics**: Duration tracking for all operations
- **Real-Time Updates**: Auto-refreshes every 10 seconds

## Quick Start

### Installation

Install dashboard dependencies:

```bash
pip install anton[dashboard]
# or for all features
pip install anton[all]
```

### Launch the Dashboard

```bash
# From your repository directory
anton dashboard

# Dashboard will be available at: http://localhost:3000
```

### Custom Port/Host

```bash
# Custom port
anton dashboard --port 8080

# Custom host (for remote access)
anton dashboard --host 0.0.0.0 --port 3000

# Specific repository
anton dashboard --path /path/to/repo
```

## Dashboard Views

### Main Dashboard

The main dashboard shows:
- **Stats Bar**: Total traces, cost, tokens, errors
- **Trace List**: Recent traces with summary metrics
- **Filters**: (Coming soon) Filter by date, agent, cost, etc.

### Trace Details

Click on any trace to see:
- **Timeline View**: Hierarchical execution flow
- **LLM Calls**: Provider, model, tokens, cost, messages
- **Tool Calls**: Tool name, parameters, results, duration
- **Error Details**: Full error messages and stack traces

## Architecture

### Components

1. **Enhanced Tracing** (`src/modular_agents/trace.py`)
   - Distributed tracing with span hierarchy
   - Token usage and cost tracking
   - JSONL file storage

2. **Trace Aggregator** (`src/modular_agents/trace_aggregator.py`)
   - Reads and parses JSONL trace files
   - Builds hierarchical timeline data
   - Aggregates metrics across traces

3. **FastAPI Server** (`src/modular_agents/dashboard_server.py`)
   - REST API endpoints for trace data
   - Serves static frontend files
   - Real-time data access

4. **Frontend** (`dashboard/index.html`)
   - Single-page application
   - Timeline visualization
   - Auto-refresh capability

### Trace Hierarchy

Traces use OpenTelemetry-style span hierarchy:

```
Task (trace_id: abc123)
├── Phase 1 (span_id: def456, parent_id: abc123)
│   ├── LLM Call (span_id: ghi789, parent_id: def456)
│   │   ├── Tool: read_file (span_id: jkl012, parent_id: ghi789)
│   │   └── Tool: grep_codebase (span_id: mno345, parent_id: ghi789)
│   └── LLM Call (span_id: pqr678, parent_id: def456)
└── Phase 2 (span_id: stu901, parent_id: abc123)
    └── ...
```

## API Endpoints

The dashboard server provides these REST endpoints:

### `GET /api/traces`

List recent traces with summary metrics.

**Query Parameters:**
- `limit` (int): Number of traces to return (default: 10, max: 100)

**Response:**
```json
{
  "count": 10,
  "traces": [
    {
      "trace_id": "abc123...",
      "start_time": "2025-11-30T10:00:00",
      "end_time": "2025-11-30T10:05:30",
      "duration_ms": 330000,
      "llm_calls": 15,
      "tool_calls": 42,
      "total_cost": 0.0234,
      "total_tokens": 12500,
      "errors": 0
    }
  ]
}
```

### `GET /api/traces/{trace_id}`

Get full trace details including hierarchical spans.

**Response:**
```json
{
  "trace_id": "abc123...",
  "start_time": "2025-11-30T10:00:00",
  "end_time": "2025-11-30T10:05:30",
  "duration_ms": 330000,
  "total_llm_calls": 15,
  "total_tool_calls": 42,
  "total_cost_usd": 0.0234,
  "total_tokens": {"input": 8500, "output": 4000, "total": 12500},
  "errors": [],
  "root_spans": [
    {
      "span_id": "def456",
      "span_type": "llm_call",
      "name": "module_database",
      "timestamp": "2025-11-30T10:00:00",
      "duration_ms": 2500,
      "success": true,
      "provider": "claude",
      "model": "claude-sonnet-3-5",
      "tokens": {"input": 1500, "output": 800, "total": 2300},
      "cost_usd": 0.0165,
      "children": [
        {
          "span_id": "jkl012",
          "span_type": "tool_call",
          "tool_name": "read_file",
          "duration_ms": 150,
          "success": true
        }
      ]
    }
  ]
}
```

### `GET /api/stats`

Get aggregate statistics across all traces.

**Response:**
```json
{
  "total_traces": 150,
  "total_llm_calls": 1250,
  "total_tool_calls": 3800,
  "total_cost_usd": 5.67,
  "total_tokens": 450000,
  "total_errors": 12,
  "avg_duration_ms": 45000
}
```

### `GET /health`

Health check endpoint.

## Trace Storage

Traces are stored in JSONL format (one JSON object per line) in `.modular-agents/traces/`:

```
your-repo/
└── .modular-agents/
    └── traces/
        ├── module_database_20251130_100000.jsonl
        ├── module_api_20251130_100030.jsonl
        └── orchestrator_20251130_095959.jsonl
```

Each line in a JSONL file is either an `LLMInteraction` or `ToolCallSpan`:

**LLMInteraction:**
```json
{
  "timestamp": "2025-11-30T10:00:00",
  "agent_name": "module_database",
  "provider": "claude",
  "model": "claude-sonnet-3-5",
  "trace_id": "abc123",
  "span_id": "def456",
  "parent_id": null,
  "span_type": "llm_call",
  "duration_ms": 2500,
  "tokens": {"input": 1500, "output": 800, "total": 2300},
  "cost_usd": 0.0165,
  "messages": [...],
  "response": "..."
}
```

**ToolCallSpan:**
```json
{
  "timestamp": "2025-11-30T10:00:01",
  "tool_name": "read_file",
  "trace_id": "abc123",
  "span_id": "jkl012",
  "parent_id": "def456",
  "span_type": "tool_call",
  "agent_name": "module_database",
  "duration_ms": 150,
  "success": true,
  "parameters": {"file_path": "/path/to/file.py"},
  "result": "file contents..."
}
```

## Cost Estimation

The dashboard estimates costs based on current (2025) pricing:

| Provider | Model | Input ($/1M tokens) | Output ($/1M tokens) |
|----------|-------|--------------------:|---------------------:|
| Claude | Sonnet 3.5/4 | $3.00 | $15.00 |
| Claude | Haiku | $0.25 | $1.25 |
| Claude | Opus | $15.00 | $75.00 |
| OpenAI | GPT-4 Turbo | $10.00 | $30.00 |
| OpenAI | GPT-4o | $5.00 | $15.00 |
| OpenAI | GPT-3.5 Turbo | $0.50 | $1.50 |

Unknown models use a default of $1.00 input / $3.00 output per 1M tokens.

## Development

### Running Locally

```bash
# Install in development mode
pip install -e ".[dashboard,dev]"

# Run the server directly
python -m modular_agents.dashboard_server

# Or via CLI
anton dashboard
```

### Customizing the Frontend

The frontend is a single HTML file at `dashboard/index.html`. Edit it to customize:
- Colors and styling
- Timeline visualization
- Metrics displayed
- Auto-refresh interval

### Adding New Endpoints

Add new endpoints in `src/modular_agents/dashboard_server.py`:

```python
@app.get("/api/custom")
async def custom_endpoint() -> dict:
    # Your logic here
    return {"data": "..."}
```

## Troubleshooting

### Dashboard won't start

**Error:** `No traces directory found`

**Solution:** Run some tasks first to generate trace data:
```bash
anton task . "Your task description"
```

### Dashboard shows no data

**Check:**
1. Traces exist: `ls .modular-agents/traces/`
2. JSONL files are valid JSON
3. Server is running on correct port: `netstat -an | grep 3000`

### Port already in use

**Error:** `Address already in use`

**Solution:** Use a different port:
```bash
anton dashboard --port 8080
```

### Can't connect from remote machine

**Solution:** Bind to all interfaces:
```bash
anton dashboard --host 0.0.0.0 --port 3000
# Then access via: http://your-server-ip:3000
```

## Future Enhancements

Planned features:
- [ ] Live streaming (WebSocket) for real-time updates
- [ ] Trace filtering and search
- [ ] Export traces to OpenTelemetry format
- [ ] Cost budget alerts
- [ ] Performance regression detection
- [ ] Trace comparison view
- [ ] Custom dashboards/widgets

## Related Documentation

- [Continuation & Autonomy](CONTINUATION.md) - Task execution features
- [Tool Calling](TOOL_CALLING.md) - Agent tool system
- [Knowledge Base](KNOWLEDGE_BASE.md) - Semantic code search

## Support

For issues or feature requests, please open an issue on GitHub with:
- Dashboard version (`anton --version`)
- Browser and version
- Steps to reproduce
- Screenshots if applicable
