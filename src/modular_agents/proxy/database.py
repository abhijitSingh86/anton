"""Database layer for proxy trace storage."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


class ProxyDatabase:
    """SQLite database for storing proxy traces."""

    def __init__(self, db_path: Path | None = None):
        if db_path is None:
            db_path = Path.home() / ".modular-agents" / "proxy_traces.db"

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        schema_path = Path(__file__).parent / "schema.sql"

        with sqlite3.connect(self.db_path) as conn:
            with open(schema_path) as f:
                conn.executescript(f.read())
            
            # Migration: Add task_id if missing
            try:
                conn.execute("ALTER TABLE traces ADD COLUMN task_id TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists

    def log_request(
        self,
        call_id: str,
        trace_id: str,
        provider: str,
        model: str,
        messages: list[dict],
        agent_name: str | None = None,
        task_id: str | None = None,
    ):
        """Log LLM request."""
        with sqlite3.connect(self.db_path) as conn:
            # Ensure trace exists
            conn.execute(
                "INSERT OR IGNORE INTO traces (trace_id, task_id) VALUES (?, ?)",
                (trace_id, task_id)
            )
            
            # Update task_id if it was NULL (for existing traces being reused, though unlikely)
            if task_id:
                conn.execute(
                    "UPDATE traces SET task_id = ? WHERE trace_id = ? AND task_id IS NULL",
                    (task_id, trace_id)
                )

            # Log call
            conn.execute("""
                INSERT INTO llm_calls (
                    call_id, trace_id, created_at, provider, model, messages, agent_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                call_id,
                trace_id,
                datetime.now().isoformat(),
                provider,
                model,
                json.dumps(messages),
                agent_name,
            ))

    def log_response(
        self,
        call_id: str,
        response: str,
        duration_ms: float,
        tokens_input: int,
        tokens_output: int,
        cost_usd: float,
        finish_reason: str | None = None,
    ):
        """Log LLM response."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE llm_calls SET
                    response = ?,
                    duration_ms = ?,
                    tokens_input = ?,
                    tokens_output = ?,
                    cost_usd = ?,
                    finish_reason = ?
                WHERE call_id = ?
            """, (response, duration_ms, tokens_input, tokens_output, cost_usd, finish_reason, call_id))

    def log_error(self, call_id: str, error: str):
        """Log LLM error."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE llm_calls SET error = ? WHERE call_id = ?",
                (error, call_id)
            )

    def get_trace(self, trace_id: str) -> dict | None:
        """Get complete trace with all calls."""
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

    def list_traces(self, limit: int = 10, task_id: str | None = None) -> list[dict]:
        """List recent traces."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            if task_id:
                traces = conn.execute(
                    "SELECT * FROM traces WHERE task_id = ? ORDER BY created_at DESC LIMIT ?",
                    (task_id, limit)
                ).fetchall()
            else:
                traces = conn.execute(
                    "SELECT * FROM traces ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            return [dict(trace) for trace in traces]

    def list_tasks(self, limit: int = 10) -> list[dict]:
        """List recent tasks."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            tasks = conn.execute("""
                SELECT task_id, MIN(created_at) as created_at, COUNT(*) as trace_count 
                FROM traces 
                WHERE task_id IS NOT NULL 
                GROUP BY task_id 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(task) for task in tasks]

    def update_trace_status(self, trace_id: str, status: str):
        """Update trace status."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE traces SET status = ? WHERE trace_id = ?",
                (status, trace_id)
            )
