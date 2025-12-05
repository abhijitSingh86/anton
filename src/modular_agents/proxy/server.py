"""LLM Proxy Server - Transparent tracing proxy for LLM calls."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .database import ProxyDatabase

app = FastAPI(title="Anton LLM Proxy")
db = ProxyDatabase()

# Current trace context (set via HTTP header)
CURRENT_TRACE_ID = None
CURRENT_TASK_ID = None

# Backend configuration (where to forward requests)
# Defaults to official APIs, can be overridden at proxy start
OPENAI_BACKEND_URL = "https://api.openai.com/v1"
CLAUDE_BACKEND_URL = "https://api.anthropic.com"


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    """Extract trace ID from header."""
    global CURRENT_TRACE_ID, CURRENT_TASK_ID
    CURRENT_TRACE_ID = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
    CURRENT_TASK_ID = request.headers.get("X-Task-ID")
    response = await call_next(request)
    return response


# ============================================================================
# Claude/Anthropic Proxy
# ============================================================================

@app.post("/v1/messages")
async def claude_messages_proxy(request: Request):
    """Proxy for Claude API - /v1/messages endpoint."""
    try:
        import anthropic
    except ImportError:
        return JSONResponse(
            status_code=500,
            content={"error": "anthropic package not installed. Install with: pip install anthropic"}
        )

    call_id = str(uuid.uuid4())
    start_time = time.time()

    # Parse request
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "unknown")
    agent_name = request.headers.get("X-Agent-Name", "unknown")

    # Log request to DB
    db.log_request(
        call_id=call_id,
        trace_id=CURRENT_TRACE_ID,
        provider="claude",
        model=model,
        messages=messages,
        agent_name=agent_name,
        task_id=CURRENT_TASK_ID,
    )

    try:
        # Forward to configured Claude backend
        client = anthropic.AsyncAnthropic(
            api_key=request.headers.get("x-api-key"),
            base_url=CLAUDE_BACKEND_URL
        )

        response = await client.messages.create(**body)

        # Calculate metrics
        duration_ms = (time.time() - start_time) * 1000
        tokens_input = response.usage.input_tokens
        tokens_output = response.usage.output_tokens
        cost_usd = estimate_cost("claude", model, tokens_input, tokens_output)

        # Extract response text
        response_text = ""
        if response.content:
            for content_block in response.content:
                if hasattr(content_block, 'text'):
                    response_text += content_block.text

        # Log response to DB
        db.log_response(
            call_id=call_id,
            response=response_text,
            duration_ms=duration_ms,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost_usd,
            finish_reason=response.stop_reason,
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

async def _openai_chat_handler(request: Request):
    """Shared handler for OpenAI chat completions."""
    try:
        import openai
    except ImportError:
        return JSONResponse(
            status_code=500,
            content={"error": "openai package not installed. Install with: pip install openai"}
        )

    call_id = str(uuid.uuid4())
    start_time = time.time()

    # Parse request
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "unknown")
    agent_name = request.headers.get("X-Agent-Name", "unknown")

    # Log request to DB
    db.log_request(
        call_id=call_id,
        trace_id=CURRENT_TRACE_ID,
        provider="openai",
        model=model,
        messages=messages,
        agent_name=agent_name,
        task_id=CURRENT_TASK_ID,
    )

    try:
        # Forward to configured OpenAI backend
        client = openai.AsyncOpenAI(
            api_key=request.headers.get("Authorization", "").replace("Bearer ", ""),
            base_url=OPENAI_BACKEND_URL
        )

        response = await client.chat.completions.create(**body)

        # Calculate metrics
        duration_ms = (time.time() - start_time) * 1000
        tokens_input = response.usage.prompt_tokens
        tokens_output = response.usage.completion_tokens
        cost_usd = estimate_cost("openai", model, tokens_input, tokens_output)

        # Extract response text
        response_text = response.choices[0].message.content if response.choices else ""

        # Log response to DB
        db.log_response(
            call_id=call_id,
            response=response_text,
            duration_ms=duration_ms,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost_usd,
            finish_reason=response.choices[0].finish_reason if response.choices else None,
        )

        # Return response to client
        return JSONResponse(response.model_dump())

    except Exception as e:
        # Log error
        db.log_error(call_id, str(e))
        raise


@app.post("/v1/chat/completions")
async def openai_chat_proxy_v1(request: Request):
    """Proxy for OpenAI API - /v1/chat/completions endpoint."""
    return await _openai_chat_handler(request)


@app.post("/chat/completions")
async def openai_chat_proxy(request: Request):
    """Proxy for OpenAI API - /chat/completions endpoint (no /v1 prefix)."""
    return await _openai_chat_handler(request)


def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD."""
    # Pricing per 1M tokens
    pricing = {
        "claude": {
            "claude-sonnet-3-5": (3.0, 15.0),
            "claude-sonnet-4": (3.0, 15.0),
            "claude-haiku": (0.25, 1.25),
        },
        "openai": {
            "gpt-4-turbo": (10.0, 30.0),
            "gpt-4o": (5.0, 15.0),
            "gpt-4": (30.0, 60.0),
        }
    }

    # Try to find pricing for this model
    provider_pricing = pricing.get(provider, {})

    # Try exact match first
    rates = provider_pricing.get(model)

    # If no exact match, try partial match
    if not rates:
        for price_model, price_rates in provider_pricing.items():
            if price_model in model:
                rates = price_rates
                break

    # Default to (0, 0) if no match found
    if not rates:
        rates = (0, 0)

    return (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000


def run_proxy(
    host: str = "localhost",
    port: int = 8001,
    openai_backend: str | None = None,
    claude_backend: str | None = None,
):
    """Run the proxy server with configurable backends.

    Args:
        host: Host to bind to
        port: Port to bind to
        openai_backend: Backend URL for OpenAI requests (default: official API)
        claude_backend: Backend URL for Claude requests (default: official API)
    """
    global OPENAI_BACKEND_URL, CLAUDE_BACKEND_URL

    # Configure backends
    if openai_backend:
        OPENAI_BACKEND_URL = openai_backend
    if claude_backend:
        CLAUDE_BACKEND_URL = claude_backend

    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_proxy()
