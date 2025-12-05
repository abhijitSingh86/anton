# Real-Time Interactive Dashboard Design

## Overview
A sophisticated, real-time debugging and observability system for the Anton multi-agent framework with graph visualization, interactive prompt editing, and execution replay capabilities.

## Architecture

### 1. Real-Time Communication Layer

**WebSocket Server** (FastAPI + WebSockets)
```
Client <--WebSocket--> Server <--File Watcher--> Trace Files
```

**Features**:
- Live trace streaming as tasks execute
- No page refresh required
- Append-only updates to UI
- File system watching for new trace entries

### 2. Graph Visualization Engine

**Technology**: HTML5 Canvas + D3.js or Cytoscape.js

**Node Types**:
- `Orchestrator` - Blue circle, central coordinator
- `LLM Call` - Purple rounded rectangle with model info
- `Tool Call` - Green hexagon with tool name
- `Sub-agent` - Orange circle with agent name
- `Response` - Gray rectangle with truncated text

**Edge Types**:
- `Request` - Solid arrow (agent → LLM/tool)
- `Response` - Dashed arrow (LLM/tool → agent)
- `Delegation` - Thick arrow (orchestrator → sub-agent)

**Layout Algorithm**:
- Hierarchical top-down layout
- Time-based horizontal positioning
- Parallel calls shown side-by-side

**Example Flow**:
```
Orchestrator
    ↓ request
  LLM (decompose)
    ↓ response
Orchestrator
    ├→ Sub-agent A    ├→ Sub-agent B
    │   ↓ request     │   ↓ request
    │  Tool (grep)    │  LLM (code)
    │   ↓ result      │   ↓ response
    │  LLM (analyze)  │  Tool (write)
    ↓                 ↓
Orchestrator (merge)
```

### 3. Interactive Debugging Features

#### A. Prompt Editor
- Click any LLM node to view full prompt
- Edit system prompt, user message, or parameters
- "Test Prompt" button to execute with new prompt
- Side-by-side comparison of original vs new response

#### B. Response Comparison
```
┌─────────────────────────────────────────┐
│ Original Response   │ New Response      │
├─────────────────────┼───────────────────┤
│ <original text>     │ <new text>        │
│                     │                   │
│ Diff highlighting: added, removed     │
└─────────────────────────────────────────┘
```

#### C. Execution Replay
- "Rerun from here" button on any node
- Creates checkpoint and re-executes downstream
- Shows both original and new execution paths
- Highlights differences in final output

### 4. Data Model

#### Trace Event Stream
```json
{
  "event_type": "llm_call|tool_call|agent_start|agent_end",
  "trace_id": "uuid",
  "span_id": "uuid",
  "parent_id": "uuid",
  "timestamp": "ISO8601",
  "agent_name": "orchestrator|module_agent",
  "data": {
    // LLM call
    "messages": [...],
    "response": "...",
    "model": "...",
    "tokens": {...},
    "cost_usd": 0.00,

    // Tool call
    "tool_name": "grep_codebase",
    "parameters": {...},
    "result": "...",
    "duration_ms": 123
  }
}
```

#### Graph Node
```json
{
  "id": "span_id",
  "type": "orchestrator|llm|tool|agent",
  "label": "Short description",
  "timestamp": "ISO8601",
  "parent": "span_id",
  "children": ["span_id"],
  "data": {
    "full_prompt": "...",
    "response": "...",
    "editable": true,
    "checkpoint_data": {...}
  }
}
```

### 5. Backend Enhancements

#### A. WebSocket Endpoint
```python
@app.websocket("/ws/traces/{trace_id}")
async def trace_stream(websocket: WebSocket, trace_id: str):
    await websocket.accept()

    # Watch trace file for updates
    async for event in watch_trace_file(trace_id):
        await websocket.send_json(event)
```

#### B. Prompt Execution API
```python
@app.post("/api/execute/llm")
async def execute_llm_prompt(
    provider: str,
    model: str,
    messages: list[dict],
    original_span_id: str
):
    # Execute LLM call with new prompt
    # Return response + comparison data
```

#### C. Checkpoint/Replay API
```python
@app.post("/api/replay/{span_id}")
async def replay_from_checkpoint(
    span_id: str,
    modifications: dict
):
    # Load execution state at span
    # Replay with modifications
    # Return new execution graph
```

### 6. Frontend Architecture

#### Component Structure
```
Dashboard (SPA)
├── GraphCanvas (main visualization)
│   ├── Node (clickable, draggable)
│   ├── Edge (animated data flow)
│   └── Controls (zoom, pan, layout)
├── DetailPanel (right sidebar)
│   ├── PromptEditor
│   ├── ResponseView
│   └── ComparisonView
├── Timeline (bottom)
│   └── EventList (chronological)
└── Toolbar (top)
    ├── TraceSelector
    ├── FilterControls
    └── ExportButton
```

#### State Management
```javascript
const state = {
  currentTrace: null,
  nodes: Map<string, Node>,
  edges: Map<string, Edge>,
  selectedNode: null,
  wsConnection: WebSocket,
  replayMode: false,
  comparisonData: null
};
```

### 7. Alternative: Proxy-Based Tracing

#### Architecture
```
Application → Proxy Server → LLM API
                ↓
            SQLite DB
                ↓
            Dashboard
```

#### Proxy Implementation
```python
class LLMProxy:
    """Transparent proxy that logs all LLM calls"""

    async def forward_request(self, provider, request):
        # Log request to DB
        trace_id = self.db.log_request(provider, request)

        # Forward to actual LLM
        response = await self.llm_client.call(provider, request)

        # Log response to DB
        self.db.log_response(trace_id, response)

        return response
```

#### Benefits
- No code modification required
- Works with any LLM provider
- Centralized logging
- Can intercept and modify requests

#### Limitations
- Misses tool calls (unless tools also proxied)
- Adds network latency
- Requires infrastructure changes

### 8. Implementation Phases

#### Phase 1: Real-Time Foundation (Week 1)
- [ ] WebSocket endpoint for live trace streaming
- [ ] File watcher for new trace entries
- [ ] Basic SPA with live updates
- [ ] Remove auto-refresh, add append-only updates

#### Phase 2: Graph Visualization (Week 2)
- [ ] Canvas-based graph renderer
- [ ] Node/edge types and styling
- [ ] Hierarchical layout algorithm
- [ ] Zoom, pan, and interaction

#### Phase 3: Interactive Debugging (Week 3)
- [ ] Prompt editor with syntax highlighting
- [ ] LLM execution API
- [ ] Response comparison view
- [ ] Diff highlighting

#### Phase 4: Execution Replay (Week 4)
- [ ] Checkpoint system
- [ ] Replay API
- [ ] Branch visualization (original vs replay)
- [ ] Result comparison

#### Phase 5: Proxy Option (Optional)
- [ ] Standalone proxy server
- [ ] SQLite trace storage
- [ ] Provider adapters (Claude, OpenAI, etc.)
- [ ] Dashboard integration

### 9. Technology Stack

**Backend**:
- FastAPI (WebSocket support)
- asyncio (file watching)
- SQLite (optional, for proxy mode)

**Frontend**:
- Vanilla JS or React (for complex state)
- D3.js or Cytoscape.js (graph visualization)
- CodeMirror (prompt editing)
- diff-match-patch (response comparison)

**Real-Time**:
- WebSockets (live updates)
- watchdog (file system monitoring)

### 10. User Workflows

#### Workflow 1: Live Debugging
1. Start task: `anton task . "your task"`
2. Open dashboard: `http://localhost:3000`
3. See graph populate in real-time as agents communicate
4. Click on LLM nodes to inspect prompts
5. Identify issues as they happen

#### Workflow 2: Prompt Optimization
1. Select completed trace
2. Click LLM node with suboptimal response
3. Edit prompt in side panel
4. Click "Test Prompt"
5. Compare original vs new response
6. If better, click "Rerun from here"
7. See new execution path

#### Workflow 3: Historical Analysis
1. Browse past traces
2. Filter by date, agent, or error
3. Visualize execution graph
4. Export problematic traces for team review

### 11. Key Features Summary

✅ **Real-Time**:
- Live graph updates via WebSocket
- No page refresh
- Append-only UI updates

✅ **Visualization**:
- Canvas-based flow diagram
- Clear agent → LLM → tool → agent paths
- Time-based layout

✅ **Interactive**:
- Edit prompts inline
- Test new prompts immediately
- Compare responses side-by-side

✅ **Debuggable**:
- Full trace visibility
- Checkpoint and replay
- Branch comparison

✅ **Scalable**:
- Optional proxy mode
- Database storage
- Works without code changes

### 12. Next Steps

1. **Immediate**: Create WebSocket endpoint and basic SPA
2. **Short-term**: Implement graph visualization
3. **Medium-term**: Add interactive debugging
4. **Long-term**: Build proxy-based alternative

---

**Questions for User**:
1. Prefer integrated approach (modify Anton) or proxy approach (separate service)?
2. Graph library preference: D3.js (flexible) vs Cytoscape.js (graph-focused)?
3. Priority: Real-time updates vs interactive debugging vs replay?
4. Need to support multiple concurrent tasks or one at a time?
