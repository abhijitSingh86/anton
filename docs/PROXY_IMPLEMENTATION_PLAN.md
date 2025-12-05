# Proxy-Based Tracing Implementation Plan

## Overview
Replace all embedded tracing code with a standalone proxy server that intercepts LLM calls transparently, keeping the Anton source code completely clean.

## Phase 1: Cleanup - Remove All Tracing Code

### Files to DELETE Entirely
```bash
rm src/modular_agents/trace.py
rm src/modular_agents/trace_aggregator.py
rm src/modular_agents/dashboard_server.py
rm -rf dashboard/
```

### Files to MODIFY (Remove Tracing Code)

**1. `src/modular_agents/runtime.py`**
- Remove lines 19: `from modular_agents.trace import init_trace_logger`
- Remove lines 129-133: Trace logger initialization
- Keep everything else intact

**2. `src/modular_agents/agents/base.py`**
- Remove trace-related imports
- Remove `_last_trace_id` and `_last_span_id` fields
- Remove trace logging in `think()` method
- Keep core agent logic

**3. `src/modular_agents/agents/module_agent.py`**
- Remove trace context passing to tool execution
- Keep tool execution logic

**4. `src/modular_agents/agents/orchestrator.py`**
- Remove trace context propagation
- Keep orchestration logic

**5. `src/modular_agents/tools/base.py`**
- Remove `ToolCallSpan` and tracing
- Remove `trace_id`, `parent_id`, `agent_name` parameters
- Keep core tool execution

**6. `pyproject.toml`**
- Remove `dashboard` optional dependencies
- Add new `proxy` optional dependencies

## Phase 2: Proxy Server Implementation

### Architecture
```
Anton (unchanged) → LLM Proxy → Actual LLM API
                         ↓
                    SQLite DB
                         ↓
                  Dashboard Server
                         ↓
                   Real-time UI
```

### Database Schema

**File: `src/modular_agents/proxy/schema.sql`**
```sql
-- Traces table (one per task execution)
CREATE TABLE traces (
    trace_id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    task_description TEXT,
    status TEXT DEFAULT 'running'  -- running, completed, failed
);

-- LLM calls table
CREATE TABLE llm_calls (
    call_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    parent_id TEXT,  -- For nested calls
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Request
    provider TEXT NOT NULL,  -- claude, openai, etc.
    model TEXT NOT NULL,
    messages JSON NOT NULL,

    -- Response
    response TEXT,
    finish_reason TEXT,

    -- Metrics
    duration_ms REAL,
    tokens_input INTEGER,
    tokens_output INTEGER,
    cost_usd REAL,

    -- Metadata
    agent_name TEXT,
    error TEXT,

    FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
);

-- Graph edges (for visualization)
CREATE TABLE call_graph (
    edge_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    from_call_id TEXT NOT NULL,
    to_call_id TEXT NOT NULL,
    edge_type TEXT,  -- request, response, delegation

    FOREIGN KEY (trace_id) REFERENCES traces(trace_id),
    FOREIGN KEY (from_call_id) REFERENCES llm_calls(call_id),
    FOREIGN KEY (to_call_id) REFERENCES llm_calls(call_id)
);

-- Indexes for performance
CREATE INDEX idx_llm_calls_trace ON llm_calls(trace_id);
CREATE INDEX idx_llm_calls_created ON llm_calls(created_at);
CREATE INDEX idx_graph_trace ON call_graph(trace_id);
```

### Proxy Server

**File: `src/modular_agents/proxy/server.py`**

```python
"""LLM Proxy Server - Transparent tracing proxy for LLM calls"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
import openai
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .database import ProxyDatabase

app = FastAPI(title="Anton LLM Proxy")
db = ProxyDatabase()

# Current trace context (set via HTTP header)
CURRENT_TRACE_ID = None


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    """Extract trace ID from header"""
    global CURRENT_TRACE_ID
    CURRENT_TRACE_ID = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
    response = await call_next(request)
    return response


# ============================================================================
# Claude/Anthropic Proxy
# ============================================================================

@app.post("/v1/messages")
async def claude_messages_proxy(request: Request):
    """Proxy for Claude API - /v1/messages endpoint"""

    call_id = str(uuid.uuid4())
    start_time = time.time()

    # Parse request
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "unknown")

    # Log request to DB
    db.log_request(
        call_id=call_id,
        trace_id=CURRENT_TRACE_ID,
        provider="claude",
        model=model,
        messages=messages
    )

    try:
        # Forward to actual Claude API
        client = anthropic.AsyncAnthropic(
            api_key=request.headers.get("x-api-key")
        )

        response = await client.messages.create(**body)

        # Calculate metrics
        duration_ms = (time.time() - start_time) * 1000
        tokens_input = response.usage.input_tokens
        tokens_output = response.usage.output_tokens
        cost_usd = estimate_cost("claude", model, tokens_input, tokens_output)

        # Log response to DB
        db.log_response(
            call_id=call_id,
            response=response.content[0].text,
            duration_ms=duration_ms,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost_usd
        )

        # Return response to client
        return JSONResponse(response.model_dump())

    except Exception as e:
        # Log error
        db.log_error(call_id, str(e))
        raise


# ============================================================================
# OpenAI Proxy
# ============================================================================

@app.post("/v1/chat/completions")
async def openai_chat_proxy(request: Request):
    """Proxy for OpenAI API - /v1/chat/completions endpoint"""

    call_id = str(uuid.uuid4())
    start_time = time.time()

    # Parse request
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "unknown")

    # Log request to DB
    db.log_request(
        call_id=call_id,
        trace_id=CURRENT_TRACE_ID,
        provider="openai",
        model=model,
        messages=messages
    )

    try:
        # Forward to actual OpenAI API
        client = openai.AsyncOpenAI(
            api_key=request.headers.get("Authorization", "").replace("Bearer ", "")
        )

        response = await client.chat.completions.create(**body)

        # Calculate metrics
        duration_ms = (time.time() - start_time) * 1000
        tokens_input = response.usage.prompt_tokens
        tokens_output = response.usage.completion_tokens
        cost_usd = estimate_cost("openai", model, tokens_input, tokens_output)

        # Log response to DB
        db.log_response(
            call_id=call_id,
            response=response.choices[0].message.content,
            duration_ms=duration_ms,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost_usd
        )

        # Return response to client
        return JSONResponse(response.model_dump())

    except Exception as e:
        # Log error
        db.log_error(call_id, str(e))
        raise


def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD"""
    # Pricing per 1M tokens
    pricing = {
        "claude": {
            "claude-sonnet-3-5": (3.0, 15.0),
            "claude-haiku": (0.25, 1.25),
        },
        "openai": {
            "gpt-4-turbo": (10.0, 30.0),
            "gpt-4o": (5.0, 15.0),
        }
    }

    rates = pricing.get(provider, {}).get(model, (0, 0))
    return (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000


def run_proxy(host: str = "localhost", port: int = 8000):
    """Run the proxy server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
```

**File: `src/modular_agents/proxy/database.py`**

```python
"""Database layer for proxy trace storage"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


class ProxyDatabase:
    """SQLite database for storing proxy traces"""

    def __init__(self, db_path: Path | None = None):
        if db_path is None:
            db_path = Path.home() / ".modular-agents" / "proxy_traces.db"

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        schema_path = Path(__file__).parent / "schema.sql"

        with sqlite3.connect(self.db_path) as conn:
            with open(schema_path) as f:
                conn.executescript(f.read())

    def log_request(
        self,
        call_id: str,
        trace_id: str,
        provider: str,
        model: str,
        messages: list[dict]
    ):
        """Log LLM request"""
        with sqlite3.connect(self.db_path) as conn:
            # Ensure trace exists
            conn.execute(
                "INSERT OR IGNORE INTO traces (trace_id) VALUES (?)",
                (trace_id,)
            )

            # Log call
            conn.execute("""
                INSERT INTO llm_calls (
                    call_id, trace_id, created_at, provider, model, messages
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                call_id,
                trace_id,
                datetime.now().isoformat(),
                provider,
                model,
                json.dumps(messages)
            ))

    def log_response(
        self,
        call_id: str,
        response: str,
        duration_ms: float,
        tokens_input: int,
        tokens_output: int,
        cost_usd: float
    ):
        """Log LLM response"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE llm_calls SET
                    response = ?,
                    duration_ms = ?,
                    tokens_input = ?,
                    tokens_output = ?,
                    cost_usd = ?
                WHERE call_id = ?
            """, (response, duration_ms, tokens_input, tokens_output, cost_usd, call_id))

    def log_error(self, call_id: str, error: str):
        """Log LLM error"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE llm_calls SET error = ? WHERE call_id = ?",
                (error, call_id)
            )

    def get_trace(self, trace_id: str) -> dict:
        """Get complete trace with all calls"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get trace info
            trace = conn.execute(
                "SELECT * FROM traces WHERE trace_id = ?",
                (trace_id,)
            ).fetchone()

            if not trace:
                return None

            # Get all calls
            calls = conn.execute(
                "SELECT * FROM llm_calls WHERE trace_id = ? ORDER BY created_at",
                (trace_id,)
            ).fetchall()

            return {
                "trace_id": trace["trace_id"],
                "created_at": trace["created_at"],
                "status": trace["status"],
                "calls": [dict(call) for call in calls]
            }
```

## Phase 3: Dashboard Updates

**File: `src/modular_agents/proxy/dashboard_server.py`**

- Read from SQLite instead of JSONL
- WebSocket support for real-time updates
- Graph visualization API

**File: `dashboard_proxy/index.html`**

- Real-time graph with D3.js
- No auto-refresh
- Interactive prompt editing
- Replay functionality

## Phase 4: Usage

### Start Proxy Server
```bash
# Terminal 1: Start proxy
anton proxy start --port 8000

# Terminal 2: Start dashboard
anton proxy dashboard --port 3000

# Terminal 3: Run task through proxy
export ANTHROPIC_BASE_URL=http://localhost:8000
anton task . "your task" --provider claude
```

### Environment Configuration
```bash
# ~/.bashrc or ~/.zshrc

# Proxy mode (traces all LLM calls)
export ANTHROPIC_BASE_URL=http://localhost:8000/v1
export OPENAI_BASE_URL=http://localhost:8000/v1

# Direct mode (no tracing, clean source)
unset ANTHROPIC_BASE_URL
unset OPENAI_BASE_URL
```

## Benefits

✅ **Clean Source Code**
- Zero tracing code in Anton
- No performance overhead
- Easy to maintain

✅ **Transparent Tracing**
- Works with existing code
- No modifications needed
- Captures all LLM calls

✅ **Flexible**
- Enable/disable via env vars
- Works with any LLM provider
- Centralized logging

✅ **Database Storage**
- SQL queries for analysis
- Efficient storage
- Easy to export

## Implementation Checklist

- [ ] Phase 1: Remove all tracing code
- [ ] Phase 2: Implement proxy server
- [ ] Phase 3: Create database layer
- [ ] Phase 4: Build dashboard
- [ ] Phase 5: Update CLI
- [ ] Phase 6: Documentation
- [ ] Phase 7: Testing

---

**Ready to implement!**
