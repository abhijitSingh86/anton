-- Traces table (one per task execution)
CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    task_id TEXT,  -- Grouping ID for multiple traces in one task
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    task_description TEXT,
    status TEXT DEFAULT 'running'  -- running, completed, failed
);

-- LLM calls table
CREATE TABLE IF NOT EXISTS llm_calls (
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
CREATE TABLE IF NOT EXISTS call_graph (
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
CREATE INDEX IF NOT EXISTS idx_llm_calls_trace ON llm_calls(trace_id);
CREATE INDEX IF NOT EXISTS idx_llm_calls_created ON llm_calls(created_at);
CREATE INDEX IF NOT EXISTS idx_graph_trace ON call_graph(trace_id);
