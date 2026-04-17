"""Switchboard MCP Server — entrypoint.

Exposes four tools:
    • route_completion
    • get_routing_status
    • set_routing_preferences
    • report_outcome
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from banner import print_banner
from config import settings
from config.settings import RoutingPreferences
from context.extractor import extract_from_messages, extract_task_hint
from context.serializer import build_handoff_messages
from context.state import ConversationState
from providers.anthropic import AnthropicProvider
from providers.health import ProviderHealthTracker
from providers.openrouter import OpenRouterProvider
from router.classifier import classify_task, get_preferred_models_for_task
from router.fallback_chain import FallbackChain
from router.rule_engine import evaluate_rules

# ── Logging setup ──────────────────────────────────────────────────

logger = logging.getLogger("switchboard")
logger.setLevel(logging.INFO)

_log_handler = logging.StreamHandler()
_log_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logger.addHandler(_log_handler)

# JSONL routing log
_ROUTING_LOG = Path.home() / ".switchboard" / "routing.log"


def _log_routing(entry: dict[str, Any]) -> None:
    """Append a JSONL line to the routing log."""
    entry["timestamp"] = time.time()
    # Ensure log file has restricted permissions (0600)
    log_exists = _ROUTING_LOG.exists()
    with open(_ROUTING_LOG, "a", encoding="utf-8") as f:
        if not log_exists:
            os.chmod(_ROUTING_LOG, 0o600)
        f.write(json.dumps(entry) + "\n")


# ── Server instance ────────────────────────────────────────────────

app = Server("switchboard")

# Shared state
_health_tracker = ProviderHealthTracker()
_fallback_chain = FallbackChain(health_tracker=_health_tracker)
_conversation_states: dict[str, ConversationState] = {}  # keyed by session id
_dry_run = False

# Provider singletons (lazy)
_provider_cache: dict[str, Any] = {}


def _get_provider(provider_name: str):
    if provider_name not in _provider_cache:
        if provider_name == "anthropic":
            _provider_cache[provider_name] = AnthropicProvider()
        elif provider_name == "openrouter":
            _provider_cache[provider_name] = OpenRouterProvider()
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
    return _provider_cache[provider_name]


# ── Tool definitions ───────────────────────────────────────────────

TOOLS = [
    Tool(
        name="route_completion",
        description=(
            "Classify the task, select the best model/provider, and return a completion. "
            "Transparently handles retries and fallback. The response includes the answer "
            "plus metadata about which model was used and why."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "description": "OpenAI-format message list",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string"},
                            "content": {"type": "string"},
                        },
                    },
                },
                "task_hint": {
                    "type": "string",
                    "description": "Optional hint about the task type",
                },
                "file_context": {
                    "type": "string",
                    "description": "Optional: comma-separated list of active file paths",
                },
                "stream": {
                    "type": "boolean",
                    "description": "Whether to stream the response",
                    "default": False,
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional session ID for context continuity",
                },
                "temperature": {
                    "type": "number",
                    "description": "Sampling temperature",
                    "default": 0.0,
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens to generate",
                },
            },
            "required": ["messages"],
        },
    ),
    Tool(
        name="get_routing_status",
        description=(
            "Return current provider health, rate limit counters, "
            "and the last 10 routing decisions with reasons."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="set_routing_preferences",
        description=(
            "Customize routing behavior at runtime. "
            "Settings persist until the server restarts."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "prefer_cheap": {
                    "type": "boolean",
                    "description": "Always pick the cheapest model",
                },
                "prefer_fast": {
                    "type": "boolean",
                    "description": "Always pick the fastest model",
                },
                "max_cost_per_request": {
                    "type": "number",
                    "description": "Maximum cost (USD) per request",
                },
                "blacklist_providers": {
                    "type": "array",
                    "description": "Provider names to exclude (e.g. ['anthropic', 'openrouter'])",
                    "items": {"type": "string"},
                },
            },
        },
    ),
    Tool(
        name="report_outcome",
        description=(
            "Report whether a routed response was successful and its quality. "
            "Used to update provider health scores for future routing."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "request_id": {
                    "type": "string",
                    "description": "The request_id from the route_completion response",
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the response was useful",
                },
                "quality_rating": {
                    "type": "integer",
                    "description": "Quality 1-5 (5 = excellent)",
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": ["request_id", "success"],
        },
    ),
]


# ── Tool handlers ──────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "route_completion":
        result = await _handle_route_completion(arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    elif name == "get_routing_status":
        result = await _handle_get_routing_status()
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    elif name == "set_routing_preferences":
        result = _handle_set_routing_preferences(arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    elif name == "report_outcome":
        result = await _handle_report_outcome(arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ── route_completion ───────────────────────────────────────────────

async def _handle_route_completion(args: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, str]] = args.get("messages", [])
    task_hint: str | None = args.get("task_hint")
    file_context: str | None = args.get("file_context")
    stream: bool = args.get("stream", False)
    session_id: str | None = args.get("session_id")
    temperature: float = args.get("temperature", 0.0)
    max_tokens: int | None = args.get("max_tokens")

    request_id = str(uuid.uuid4())[:8]

    if not messages:
        return {"error": "messages is required", "request_id": request_id}

    # ── Build / update conversation state ──────────────────────────
    state = extract_from_messages(messages)
    if file_context:
        state.active_files = [f.strip() for f in file_context.split(",")]
    if session_id:
        _conversation_states[session_id] = state

    # ── Estimate context size (rough token count) ──────────────────
    context_tokens = _estimate_tokens(messages)

    # ── Layer 1: Rule engine ───────────────────────────────────────
    rule_result = await evaluate_rules(
        messages,
        task_hint=task_hint,
        context_size=context_tokens,
    )

    selected_model: dict[str, Any] | None = None
    routing_reason = ""

    if rule_result.conclusive:
        model_def = settings.get_model_by_id(rule_result.model_id or "")
        if model_def:
            selected_model = model_def
            routing_reason = rule_result.reason
            logger.info("Rule engine selected: %s — %s", model_def["id"], routing_reason)

    # ── Layer 2: Classifier (if rules were inconclusive) ──────────
    if selected_model is None:
        task_description = task_hint or extract_task_hint(messages)
        file_ext = _extract_file_ext(file_context or "")
        task_category = classify_task(
            task_description,
            conversation_history=messages,
            file_extension=file_ext,
        )
        routing_reason = f"classifier: {task_category}"

        preferred = get_preferred_models_for_task(task_category)
        for m in preferred:
            provider = m.get("provider", "")
            if provider in settings.prefs.blacklist_providers:
                continue
            if _health_tracker.is_degraded(provider):
                continue
            selected_model = m
            break

        # Fallback if classifier found nothing available
        if selected_model is None:
            selected_model = _fallback_chain.get_next(task_category=task_category)
            if selected_model:
                routing_reason += f" (fallback: {selected_model['id']})"

    # Final fallback
    if selected_model is None:
        selected_model = _fallback_chain.get_next()
        routing_reason = "final fallback"

    if selected_model is None:
        return {
            "error": "No available models. All providers are degraded or unavailable.",
            "request_id": request_id,
        }

    # ── Dry run ────────────────────────────────────────────────────
    if _dry_run:
        return {
            "request_id": request_id,
            "dry_run": True,
            "would_use_model": selected_model["id"],
            "would_use_provider": selected_model["provider"],
            "routing_reason": routing_reason,
            "estimated_context_tokens": context_tokens,
        }

    # ── Build messages (use handoff if we have accumulated state) ──
    # Try with context handoff — serialize the state into a compact prompt
    handoff_messages = build_handoff_messages(state)

    # ── Execute with fallback chain ────────────────────────────────
    tried_models: set[str] = set()
    last_error: str = ""

    # Build candidate list: selected model first, then fallback chain
    candidates = [selected_model]
    fallback_models = _fallback_chain.get_all_available()
    for fm in fallback_models:
        if fm["id"] not in tried_models and fm["id"] != selected_model["id"]:
            candidates.append(fm)

    for candidate in candidates:
        candidate_id = candidate["id"]
        tried_models.add(candidate_id)

        try:
            candidate_provider = _get_provider(candidate["provider"])

            if stream:
                return await _handle_stream_response(
                    candidate_provider, candidate_id, handoff_messages,
                    temperature, max_tokens, request_id, routing_reason,
                    selected_model["provider"],
                )

            result = await asyncio.wait_for(
                candidate_provider.chat_complete(
                    messages=handoff_messages,
                    model=candidate_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                timeout=180.0,
            )

            # Success — log and return
            _log_routing({
                "request_id": request_id,
                "model_used": candidate_id,
                "provider": candidate["provider"],
                "routing_reason": routing_reason,
                "context_tokens": context_tokens,
                "success": True,
            })

            return {
                "request_id": request_id,
                "response": result["text"],
                "model_used": candidate_id,
                "provider": candidate["provider"],
                "routing_reason": routing_reason,
                "usage": result.get("usage", {}),
            }

        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "Model %s failed: %s — trying next", candidate_id, _sanitize_error(last_error)
            )
            await _health_tracker.record_error(candidate["provider"])
            routing_reason = f"fallback from {candidate_id}: {_sanitize_error(last_error)[:100]}"

    # All candidates failed
    sanitized_last_error = _sanitize_error(last_error)
    _log_routing({
        "request_id": request_id,
        "model_used": selected_model["id"],
        "provider": selected_model["provider"],
        "routing_reason": routing_reason,
        "context_tokens": context_tokens,
        "success": False,
        "error": sanitized_last_error,
    })

    return {
        "request_id": request_id,
        "error": f"All candidates failed. Last error: {sanitized_last_error}",
        "model_used": selected_model["id"],
        "provider": selected_model["provider"],
        "routing_reason": routing_reason,
    }


async def _handle_stream_response(
    provider,
    model_id: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int | None,
    request_id: str,
    routing_reason: str,
    original_provider: str,
) -> dict[str, Any]:
    """Stream the response and return a special object.

    Note: MCP's current tool protocol doesn't natively support streaming
    tool responses chunk-by-chunk. We collect the full stream and return
    it as a single response, but the streaming happens internally.
    """
    chunks: list[str] = []
    async for chunk in provider.chat_complete(
        messages=messages,
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    ):
        chunks.append(chunk)

    full_text = "".join(chunks)

    _log_routing({
        "request_id": request_id,
        "model_used": model_id,
        "provider": original_provider,
        "routing_reason": routing_reason,
        "success": True,
        "streamed": True,
    })

    return {
        "request_id": request_id,
        "response": full_text,
        "model_used": model_id,
        "provider": original_provider,
        "routing_reason": routing_reason,
        "streamed": True,
    }


# ── get_routing_status ─────────────────────────────────────────────

async def _handle_get_routing_status() -> dict[str, Any]:
    health = await _health_tracker.get_all_status()
    routing_log = _health_tracker.get_routing_log(limit=10)
    available_models = _fallback_chain.get_all_available()

    return {
        "provider_health": health,
        "available_models": [
            {"id": m["id"], "provider": m["provider"], "tier": m.get("tier", "paid")}
            for m in available_models
        ],
        "last_routing_decisions": routing_log,
        "preferences": settings.prefs.to_dict(),
        "dry_run": _dry_run,
    }


# ── set_routing_preferences ────────────────────────────────────────

def _handle_set_routing_preferences(args: dict[str, Any]) -> dict[str, Any]:
    new_prefs = RoutingPreferences.from_dict(args)
    settings.prefs.prefer_cheap = new_prefs.prefer_cheap
    settings.prefs.prefer_fast = new_prefs.prefer_fast
    settings.prefs.max_cost_per_request = new_prefs.max_cost_per_request
    settings.prefs.blacklist_providers = new_prefs.blacklist_providers

    logger.info("Preferences updated: %s", settings.prefs.to_dict())
    return {
        "status": "ok",
        "preferences": settings.prefs.to_dict(),
    }


# ── report_outcome ─────────────────────────────────────────────────

async def _handle_report_outcome(args: dict[str, Any]) -> dict[str, Any]:
    request_id = args.get("request_id", "")
    success = args.get("success", True)
    quality = args.get("quality_rating", 3)

    # Find the log entry and update provider health
    log_entries = _health_tracker.get_routing_log(limit=50)
    for entry in reversed(log_entries):
        if entry.get("request_id") == request_id:
            provider = entry.get("provider", "")
            if provider and not success:
                await _health_tracker.record_error(provider)
            break

    _log_routing({
        "request_id": request_id,
        "outcome_reported": True,
        "success": success,
        "quality_rating": quality,
    })

    return {
        "status": "ok",
        "request_id": request_id,
        "success": success,
        "quality_rating": quality,
    }


# ── Helpers ────────────────────────────────────────────────────────

def _estimate_tokens(messages: list[dict]) -> int:
    """Rough estimate: ~4 chars per token for English text."""
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return total_chars // 4


def _extract_file_ext(file_context: str) -> str:
    """Extract a file extension from the file context string."""
    if not file_context:
        return ""
    parts = file_context.split(",")
    if parts:
        path = parts[0].strip()
        if "." in path:
            return "." + path.rsplit(".", 1)[-1]
    return ""


def _sanitize_error(error: str) -> str:
    """Mask sensitive tokens like API keys in error messages."""
    # Mask common API key patterns (sk-...)
    return re.sub(r"sk-[a-zA-Z0-9\-]{10,}", "sk-REDACTED", error)


# ── CLI entry point ────────────────────────────────────────────────

def main() -> None:
    print_banner()

    parser = argparse.ArgumentParser(description="Switchboard MCP Server")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show routing decisions without making API calls",
    )
    args = parser.parse_args()

    global _dry_run
    _dry_run = args.dry_run

    logger.info("Starting Switchboard MCP Server (dry_run=%s)", _dry_run)

    async def run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options(),
            )

    asyncio.run(run())


if __name__ == "__main__":
    main()
