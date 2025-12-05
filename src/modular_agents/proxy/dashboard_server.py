"""Dashboard server for proxy traces with real-time updates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .database import ProxyDatabase

app = FastAPI(title="Anton Proxy Dashboard")
db = ProxyDatabase()


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


@app.get("/")
async def dashboard_home():
    """Serve the dashboard HTML."""
    dashboard_html = Path(__file__).parent / "dashboard.html"
    if dashboard_html.exists():
        return FileResponse(dashboard_html)
    return HTMLResponse("<h1>Dashboard UI not yet implemented</h1>")


@app.get("/api/traces")
async def list_traces(limit: int = 10, task_id: str = None):
    """List recent traces."""
    try:
        traces = db.list_traces(limit=limit, task_id=task_id)
        return {"traces": traces}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list traces: {str(e)}")


@app.get("/api/tasks")
async def list_tasks(limit: int = 10):
    """List recent tasks."""
    try:
        tasks = db.list_tasks(limit=limit)
        return {"tasks": tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")


@app.get("/api/traces/{trace_id}")
async def get_trace(trace_id: str):
    """Get a specific trace with all calls."""
    try:
        trace = db.get_trace(trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail="Trace not found")
        return trace
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trace: {str(e)}")


@app.websocket("/ws/traces")
async def websocket_traces(websocket: WebSocket):
    """WebSocket endpoint for real-time trace updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and receive any messages
            data = await websocket.receive_text()
            # Echo back for now
            await websocket.send_json({"status": "connected"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


def run_dashboard(host: str = "localhost", port: int = 3000):
    """Run the dashboard server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_dashboard()
