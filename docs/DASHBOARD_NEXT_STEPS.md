# Real-Time Dashboard - Implementation Plan

## Current Status

### ✅ Completed (Phase 0)
1. **Basic Dashboard** with REST API (`dashboard/index.html`, `dashboard_server.py`)
   - Task listing with metrics
   - Hierarchical trace display
   - Basic UI with auto-refresh

2. **Backward Compatibility** (`trace_aggregator.py`)
   - Handles old and new trace formats
   - Deterministic UUID generation for old traces

3. **Trace Infrastructure** (`trace.py`)
   - Distributed tracing with span hierarchy
   - LLM and tool call logging
   - Cost and token tracking

### 🚧 Next: Enhanced Real-Time Dashboard (Standalone)

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│  Anton Task Execution                                │
│  └── Writes to .modular-agents/traces/*.jsonl       │
└───────────────────────┬─────────────────────────────┘
                        │
                        ↓ (file watch)
┌─────────────────────────────────────────────────────┐
│  Dashboard Server (Separate Service)                 │
│  ├── FastAPI + WebSockets                            │
│  ├── File Watcher (watchdog)                         │
│  ├── LLM Execution API                               │
│  └── Checkpoint/Replay Engine                        │
└───────────────────────┬─────────────────────────────┘
                        │
                        ↓ (WebSocket)
┌─────────────────────────────────────────────────────┐
│  Frontend SPA                                        │
│  ├── D3.js Graph Visualization                       │
│  ├── Prompt Editor (CodeMirror)                      │
│  ├── Response Diff View                              │
│  └── Replay Controls                                 │
└─────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Real-Time Foundation (Current Priority)

**Files to Create:**
- `src/modular_agents/realtime_dashboard_server.py` - New standalone server
- `dashboard_v2/index.html` - New SPA (no auto-refresh)
- `dashboard_v2/js/websocket.js` - WebSocket client
- `dashboard_v2/js/state.js` - Client-side state management

**Features:**
1. WebSocket endpoint: `/ws/trace/{trace_id}`
2. File watcher using `watchdog` library
3. Stream trace events as they're written
4. Client appends data without page refresh

**Implementation:**
```python
# realtime_dashboard_server.py
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class TraceFileWatcher(FileSystemEventHandler):
    def __init__(self, websocket_manager):
        self.ws_manager = websocket_manager

    def on_modified(self, event):
        if event.src_path.endswith('.jsonl'):
            # Read new lines and broadcast via WebSocket
            new_events = read_new_lines(event.src_path)
            await self.ws_manager.broadcast(new_events)

@app.websocket("/ws/trace/{trace_id}")
async def trace_stream(websocket: WebSocket, trace_id: str):
    await websocket.accept()
    # Send existing trace data
    # Then stream new events as they arrive
```

**Client-Side:**
```javascript
// websocket.js
const ws = new WebSocket(`ws://localhost:3000/ws/trace/${traceId}`);

ws.onmessage = (event) => {
    const traceEvent = JSON.parse(event.data);
    state.addEvent(traceEvent);  // Append to state
    graph.addNode(traceEvent);    // Update graph
    // NO page refresh!
};
```

---

### Phase 2: Graph Visualization with D3.js

**Files to Create:**
- `dashboard_v2/js/graph.js` - D3.js graph renderer
- `dashboard_v2/css/graph.css` - Graph styling

**Node Types:**
```javascript
const NODE_TYPES = {
    orchestrator: { shape: 'circle', color: '#4A90E2', size: 40 },
    llm: { shape: 'rect', color: '#9B59B6', size: 60 },
    tool: { shape: 'hexagon', color: '#2ECC71', size: 50 },
    agent: { shape: 'circle', color: '#E67E22', size: 35 }
};
```

**Graph Layout:**
```javascript
// Hierarchical force-directed layout
const simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).distance(100))
    .force('charge', d3.forceManyBody().strength(-300))
    .force('x', d3.forceX().x(d => d.timestamp * scale))  // Time-based
    .force('y', d3.forceY().y(d => d.level * 100));       // Hierarchy
```

**Interactive Features:**
- Zoom/pan with d3.zoom()
- Click node to show details
- Hover to highlight path
- Drag nodes to rearrange

---

### Phase 3: Interactive Prompt Editor

**Files to Create:**
- `dashboard_v2/js/prompt-editor.js` - Editor component
- `dashboard_v2/js/diff-viewer.js` - Response comparison

**Server Endpoint:**
```python
@app.post("/api/execute/llm")
async def execute_llm_prompt(
    provider: str,
    model: str,
    messages: list[dict],
    original_span_id: str,
    trace_id: str
):
    """Execute LLM with modified prompt, return comparison data"""

    # Call LLM API directly
    from anthropic import Anthropic
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = await client.messages.create(
        model=model,
        messages=messages
    )

    # Get original response for comparison
    original = get_span(trace_id, original_span_id)

    return {
        "original": original["response"],
        "new": response.content[0].text,
        "diff": compute_diff(original["response"], response.content[0].text),
        "new_span_id": str(uuid.uuid4())
    }
```

**UI Component:**
```javascript
// prompt-editor.js
class PromptEditor {
    constructor(container) {
        this.editor = CodeMirror(container, {
            mode: 'markdown',
            theme: 'monokai',
            lineNumbers: true
        });
    }

    async testPrompt() {
        const newPrompt = this.editor.getValue();
        const response = await fetch('/api/execute/llm', {
            method: 'POST',
            body: JSON.stringify({
                messages: [{ role: 'user', content: newPrompt }],
                original_span_id: this.currentSpanId
            })
        });

        const result = await response.json();
        this.showComparison(result.original, result.new, result.diff);
    }
}
```

---

### Phase 4: Checkpoint & Replay System

**Files to Create:**
- `src/modular_agents/checkpoint.py` - Checkpoint storage
- `src/modular_agents/replay_engine.py` - Re-execution logic

**Checkpoint Storage:**
```python
@dataclass
class Checkpoint:
    trace_id: str
    span_id: str
    timestamp: datetime

    # Execution state
    task_description: str
    repo_knowledge: dict
    module_states: dict[str, dict]

    # Modifications
    modified_prompts: dict[str, str]  # span_id -> new_prompt

    def save(self, db_path: Path):
        """Save checkpoint to SQLite"""

    @classmethod
    def load(cls, checkpoint_id: str) -> "Checkpoint":
        """Load checkpoint from storage"""
```

**Replay Endpoint:**
```python
@app.post("/api/replay/{span_id}")
async def replay_from_checkpoint(
    span_id: str,
    modifications: dict
):
    """Replay execution from a specific point"""

    # Load checkpoint
    checkpoint = Checkpoint.load_from_span(span_id)

    # Apply modifications
    checkpoint.apply_modifications(modifications)

    # Re-execute from this point
    new_trace_id = await replay_engine.execute(checkpoint)

    return {
        "original_trace_id": checkpoint.trace_id,
        "new_trace_id": new_trace_id,
        "comparison_url": f"/compare/{checkpoint.trace_id}/{new_trace_id}"
    }
```

---

## File Structure

```
/home/box/anton/
├── src/modular_agents/
│   ├── realtime_dashboard_server.py   (NEW - WebSocket server)
│   ├── checkpoint.py                  (NEW - Checkpoint system)
│   └── replay_engine.py               (NEW - Replay logic)
│
├── dashboard_v2/                      (NEW - Real-time SPA)
│   ├── index.html
│   ├── css/
│   │   ├── main.css
│   │   └── graph.css
│   ├── js/
│   │   ├── websocket.js              (WebSocket client)
│   │   ├── state.js                  (State management)
│   │   ├── graph.js                  (D3.js visualization)
│   │   ├── prompt-editor.js          (Interactive editor)
│   │   ├── diff-viewer.js            (Response comparison)
│   │   └── replay-controls.js        (Checkpoint UI)
│   └── lib/
│       ├── d3.v7.min.js
│       ├── codemirror.min.js
│       └── diff-match-patch.js
│
└── docs/
    ├── REALTIME_DASHBOARD_DESIGN.md   (Architecture)
    └── DASHBOARD_NEXT_STEPS.md        (This file)
```

## Dependencies to Add

```toml
[project.optional-dependencies]
realtime_dashboard = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "websockets>=12.0",
    "watchdog>=3.0.0",        # File system monitoring
    "anthropic>=0.18.0",      # For LLM re-execution
    "openai>=1.0.0",          # For OpenAI support
    "diff-match-patch>=20200713",  # Text diffing
]
```

## CLI Integration

```bash
# Start enhanced real-time dashboard
anton dashboard-v2 --path /project/path

# With specific trace
anton dashboard-v2 --path /project/path --trace abc-123

# Enable replay mode
anton dashboard-v2 --path /project/path --allow-replay
```

## Testing Plan

### Manual Testing
1. **Real-time Updates**:
   - Start dashboard
   - Run `anton task . "test task"`
   - Verify graph populates live without refresh

2. **Prompt Editing**:
   - Click LLM node
   - Edit prompt
   - Click "Test Prompt"
   - Verify response comparison works

3. **Replay**:
   - Select LLM node
   - Click "Rerun from here"
   - Verify new branch appears in graph

### Automated Testing
```python
# tests/test_realtime_dashboard.py
async def test_websocket_streaming():
    async with AsyncClient(app) as client:
        async with client.websocket_connect("/ws/trace/test-123") as ws:
            # Write to trace file
            write_trace_event(...)

            # Verify WebSocket receives event
            data = await ws.receive_json()
            assert data["span_id"] == "expected-id"
```

## Development Timeline

**Week 1**: Real-time foundation + basic graph
**Week 2**: Complete graph visualization + interactions
**Week 3**: Prompt editor + LLM execution API
**Week 4**: Checkpoint system + replay functionality

## Current Action Items

1. ✅ Design document created
2. ✅ Implementation plan documented
3. ⏭️ Create realtime_dashboard_server.py with WebSocket support
4. ⏭️ Build basic SPA with D3.js graph
5. ⏭️ Add file watcher for live updates
6. ⏭️ Implement prompt editor
7. ⏭️ Add checkpoint and replay

---

**Ready to proceed with implementation!**

The existing basic dashboard will remain functional while we build the enhanced version in parallel.
