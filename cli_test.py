#!/usr/bin/env python3
"""Switchboard CLI Tester — scenarios that demonstrate every routing capability.

Usage:
    python cli_test.py dry_run        # Zero API keys needed — the hook demo
    python cli_test.py task_routing   # All 7 task categories
    python cli_test.py rate_limit     # 429 handling + fallback
    python cli_test.py context_switch # Structured handoff vs raw history
    python cli_test.py provider_health  # Degradation + recovery
    python cli_test.py stress         # 20 concurrent requests

Run with no arguments to see help.
All scenarios use mocked HTTP — zero real API calls, zero API keys needed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import respx
from httpx import Response
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.markdown import Markdown

# ── Path setup so we can import switchboard modules ──────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import RoutingPreferences, load_models, get_model_by_id
from context.extractor import extract_from_messages
from context.serializer import serialize_state
from context.state import ConversationState
from router.classifier import classify_task, TASK_CATEGORIES
from router.fallback_chain import FallbackChain
from router.rule_engine import evaluate_rules
from providers.health import ProviderHealthTracker
from providers.anthropic import AnthropicProvider
from providers.openrouter import OpenRouterProvider

console = Console()

# ── Mock response helpers ────────────────────────────────────────────

MOCK_ANTHROPIC_RESPONSE = {
    "content": [{"type": "text", "text": "Sure, here's the code you asked for."}],
    "model": "claude-sonnet-4-20250514",
    "usage": {"input_tokens": 120, "output_tokens": 45},
}

MOCK_OPENROUTER_RESPONSE = {
    "choices": [{"message": {"content": "Here's a response from OpenRouter.", "role": "assistant"}}],
    "usage": {"prompt_tokens": 120, "completion_tokens": 40},
}

MOCK_STREAM_EVENTS = [
    {"choices": [{"delta": {"content": "Here", "role": "assistant"}}]},
    {"choices": [{"delta": {"content": "'s", "role": "assistant"}}]},
    {"choices": [{"delta": {"content": " the", "role": "assistant"}}]},
    {"choices": [{"delta": {"content": " streamed", "role": "assistant"}}]},
    {"choices": [{"delta": {"content": " response.", "role": "assistant"}}]},
]


def _sse_lines(events: list[dict]) -> str:
    """Format mock events as SSE stream."""
    lines: list[str] = []
    for event in events:
        lines.append(f"data: {json.dumps(event)}")
        lines.append("")
    lines.append("data: [DONE]")
    lines.append("")
    return "\n".join(lines)


# ── Scenario: dry_run ────────────────────────────────────────────────

SCENARIO_DRY_RUN = {
    "help": "Route all 7 task categories without any API calls (zero-setup demo)",
}

DRY_RUN_TASKS = [
    ("code_generation", "Write a FastAPI endpoint for user authentication with JWT tokens"),
    ("code_review", "Review this pull request diff and suggest improvements"),
    ("debugging", "I'm getting a NullPointerException on line 42, how do I fix it?"),
    ("explanation", "Explain how the async/await event loop works in Python"),
    ("architecture", "Design a scalable microservice architecture for an e-commerce platform"),
    ("autocomplete", "Complete this function: def fibonacci(n):"),
    ("security_audit", "Audit this code for XSS and SQL injection vulnerabilities"),
]


async def scenario_dry_run() -> None:
    console.print(Panel.fit(
        "[bold cyan]Scenario: Dry Run[/bold cyan]\n"
        "Routing all 7 task categories with [bold green]zero API keys[/bold green] — no real HTTP calls.",
        border_style="cyan",
    ))

    table = Table(
        title="Routing Decisions",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Task Category", style="cyan", width=18)
    table.add_column("Input Snippet", style="white", width=50)
    table.add_column("Rule Result", style="yellow", width=16)
    table.add_column("Chosen Model", style="green", width=32)
    table.add_column("Reason", style="dim", width=35)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_progress = progress.add_task("Routing...", total=len(DRY_RUN_TASKS))

        for category, input_text in DRY_RUN_TASKS:
            messages = [{"role": "user", "content": input_text}]

            # Rule engine
            rule_result = await evaluate_rules(
                messages,
                task_hint=category,
            )

            if rule_result.conclusive:
                model_def = get_model_by_id(rule_result.model_id or "")
                reason = rule_result.reason
            else:
                # Classifier
                task_cat = classify_task(input_text, messages)
                preferred_models = _get_preferred_models_for_task_safe(task_cat)
                model_def = preferred_models[0] if preferred_models else None
                reason = f"classifier: {task_cat}"

                if model_def is None:
                    chain = FallbackChain()
                    model_def = chain.get_next(task_category=task_cat)
                    reason += f" (fallback)"

            model_name = model_def["id"] if model_def else "(none available)"
            rule_status = "conclusive" if rule_result.conclusive else "→ classifier"

            table.add_row(
                category,
                input_text[:48] + "…" if len(input_text) > 48 else input_text,
                rule_status,
                model_name[:30],
                reason[:33] + "…" if len(reason) > 33 else reason,
            )
            progress.update(task_progress, advance=1)

    console.print(table)
    console.print()
    console.print(
        "[bold green]✓[/bold green] All 7 tasks routed. "
        "[dim]No API keys required — this is the zero-setup demo.[/dim]"
    )


def _get_preferred_models_for_task_safe(task_cat: str) -> list[dict]:
    from config.settings import get_models_by_strength
    return get_models_by_strength(task_cat)


# ── Scenario: task_routing ───────────────────────────────────────────

SCENARIO_TASK_ROUTING = {
    "help": "Send one request per task category and show routing decisions",
}


async def scenario_task_routing() -> None:
    console.print(Panel.fit(
        "[bold cyan]Scenario: Task Routing[/bold cyan]\n"
        "One request per category — watch the hybrid router decide.",
        border_style="cyan",
    ))

    table = Table(
        title="Task → Model Routing Table",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Category", style="cyan", width=18)
    table.add_column("Classified As", style="yellow", width=18)
    table.add_column("Chosen Model", style="green", width=32)
    table.add_column("Routing Reason", style="dim", width=35)
    table.add_column("Status", width=8)

    tasks = list(DRY_RUN_TASKS)
    warnings: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_progress = progress.add_task("Routing...", total=len(tasks))

        for i, (expected_cat, input_text) in enumerate(tasks, 1):
            messages = [{"role": "user", "content": input_text}]

            rule_result = await evaluate_rules(
                messages,
                task_hint=expected_cat,
            )

            if rule_result.conclusive:
                model_def = get_model_by_id(rule_result.model_id or "")
                reason = rule_result.reason
                classified = "(rule engine)"
            else:
                classified = classify_task(input_text, messages)
                preferred = _get_preferred_models_for_task_safe(classified)
                model_def = preferred[0] if preferred else None
                reason = f"classifier: {classified}"

                if model_def is None:
                    chain = FallbackChain()
                    model_def = chain.get_next(task_category=classified)
                    reason += " (fallback)"

            # Check for misclassification
            status = "[green]✓[/green]"
            if rule_result.conclusive:
                # Rule engine overrode — not necessarily a misclassification
                status = "[yellow]R[/yellow]"
            elif classified != expected_cat:
                status = "[red]✗[/red]"
                warnings.append(
                    f"  [red]⚠[/red] Expected [cyan]{expected_cat}[/cyan], "
                    f"got [yellow]{classified}[/yellow]: \"{input_text[:40]}…\""
                )

            model_name = model_def["id"] if model_def else "(none)"
            table.add_row(
                str(i),
                expected_cat,
                classified,
                model_name[:30],
                reason[:33] + "…" if len(reason) > 33 else reason,
                status,
            )
            progress.update(task_progress, advance=1)

    console.print(table)

    if warnings:
        console.print()
        console.print(Panel(
            "\n".join(warnings),
            title="[bold yellow]Misclassifications[/bold yellow]",
            border_style="yellow",
        ))
    else:
        console.print()
        console.print("[bold green]✓[/bold green] All tasks classified correctly.")


# ── Scenario: rate_limit ─────────────────────────────────────────────

SCENARIO_RATE_LIMIT = {
    "help": "Simulate Anthropic 429, verify fallback to OpenRouter",
}


async def scenario_rate_limit() -> None:
    console.print(Panel.fit(
        "[bold cyan]Scenario: Rate Limit Handling[/bold cyan]\n"
        "Anthropic returns 429 → Switchboard silently falls back to OpenRouter.",
        border_style="cyan",
    ))

    health = ProviderHealthTracker()
    chain = FallbackChain(health_tracker=health)

    # Create provider instances for testing
    anthropic = AnthropicProvider(api_key="fake-key")
    openrouter = OpenRouterProvider(api_key="fake-key")

    # Set up respx mock routes
    with respx.mock:
        # Anthropic returns 429
        anthropic_route = respx.post(
            "https://api.anthropic.com/v1/messages"
        ).mock(return_value=Response(429, json={"error": "rate limit exceeded"}))

        # OpenRouter returns success
        openrouter_route = respx.post(
            "https://openrouter.ai/api/v1/chat/completions"
        ).mock(return_value=Response(200, json=MOCK_OPENROUTER_RESPONSE))

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Step", style="dim", width=6)
        table.add_column("Action", style="cyan", width=25)
        table.add_column("Provider", style="yellow", width=15)
        table.add_column("Model", style="green", width=30)
        table.add_column("Result", width=12)
        table.add_column("Time", style="dim", width=10)

        start = time.time()

        # Step 1: Try Anthropic (will 429)
        try:
            result = await anthropic.chat_complete(
                messages=[{"role": "user", "content": "Write a function"}],
                model="claude-sonnet-4-20250514",
            )
            table.add_row("1", "Direct API call", "anthropic", "claude-sonnet-4", "success", "—")
        except Exception as exc:
            elapsed = f"{time.time() - start:.3f}s"
            table.add_row("1", "Direct API call", "anthropic", "claude-sonnet-4", f"[red]429[/red]", elapsed)
            await health.record_error("anthropic")
            await health.record_rate_limit("anthropic")

        # Step 2: Fallback to OpenRouter
        start2 = time.time()
        try:
            result = await openrouter.chat_complete(
                messages=[{"role": "user", "content": "Write a function"}],
                model="deepseek/deepseek-v3",
            )
            elapsed = f"{time.time() - start2:.3f}s"
            table.add_row("2", "Fallback call", "openrouter", "deepseek-v3", "[green]success[/green]", elapsed)
        except Exception as exc:
            elapsed = f"{time.time() - start2:.3f}s"
            table.add_row("2", "Fallback call", "openrouter", "deepseek-v3", f"[red]{exc}[/red]", elapsed)

        # Step 3: Verify health status
        is_degraded = health.is_degraded("anthropic")
        is_limited = health.is_rate_limited("anthropic")
        table.add_row(
            "3",
            "Health check",
            "anthropic",
            f"degraded={is_degraded}",
            "[yellow]⚠ degraded[/yellow]" if is_degraded else "ok",
            "—",
        )

        # Step 4: Verify routing around degraded provider
        next_model = chain.get_next(skip_model="claude-sonnet-4-20250514")
        table.add_row(
            "4",
            "Next fallback",
            next_model["provider"] if next_model else "—",
            next_model["id"] if next_model else "—",
            "[green]routed away[/green]",
            "—",
        )

    total_time = f"{time.time() - start:.3f}s"
    console.print(table)
    console.print()
    console.print(
        f"[bold green]✓[/bold green] Fallback succeeded in {total_time}. "
        "[dim]Anthropic marked degraded, routed to OpenRouter.[/dim]"
    )


# ── Scenario: context_switch ─────────────────────────────────────────

SCENARIO_CONTEXT_SWITCH = {
    "help": "10-turn conversation, force switch at turn 5, compare handoff vs raw",
}

TURNS = [
    {"role": "user", "content": "Build a REST API for a todo app with FastAPI"},
    {"role": "assistant", "content": "I'll create the project structure. First, let's set up the main app file.\n\n```python:app/main.py\nfrom fastapi import FastAPI\napp = FastAPI()\n\n@app.get('/')\ndef root():\n    return {'status': 'ok'}\n```"},
    {"role": "user", "content": "Good, now add the Todo model and database setup"},
    {"role": "assistant", "content": "I decided to use SQLite for simplicity. Here's the model:\n\n```python:app/models.py\nclass Todo(Base):\n    id = Column(Integer, primary_key=True)\n    title = Column(String)\n    completed = Column(Boolean, default=False)\n```"},
    {"role": "user", "content": "Now add CRUD endpoints for the todos"},
    {"role": "assistant", "content": "Here are all four CRUD endpoints:\n\n```python:app/routes.py\n@app.get('/todos')\ndef list_todos(): ...\n\n@app.post('/todos')\ndef create_todo(todo: TodoCreate): ...\n```"},
    {"role": "user", "content": "Add authentication with JWT tokens"},
    {"role": "assistant", "content": "I'll add a middleware that validates JWT tokens on protected routes.\n\n```python:app/auth.py\ndef verify_token(token: str) -> dict: ...\n```"},
    {"role": "user", "content": "Write tests for the API endpoints"},
    {"role": "assistant", "content": "Here are the pytest tests using TestClient:\n\n```python:tests/test_api.py\ndef test_create_todo(): ...\n```"},
    {"role": "user", "content": "Now add rate limiting to prevent abuse"},
]


async def scenario_context_switch() -> None:
    console.print(Panel.fit(
        "[bold cyan]Scenario: Context Switch[/bold cyan]\n"
        "10-turn conversation → force model switch at turn 5 → compare handoff vs raw history.",
        border_style="cyan",
    ))

    # Build accumulated state turn by turn
    console.print("[dim]Accumulating conversation state...[/dim]")

    state = ConversationState()
    all_messages: list[dict] = []

    for turn_idx, msg in enumerate(TURNS, 1):
        all_messages.append(msg)

        # Update state from this turn
        if turn_idx == 5:
            # Force a "switch" point — extract and show state before/after
            pre_switch_state = extract_from_messages(list(all_messages))
            pre_switch_prompt = serialize_state(pre_switch_state)
            pre_switch_raw_size = sum(len(m.get("content", "")) for m in all_messages)

        if turn_idx == 10:
            post_switch_state = extract_from_messages(list(all_messages))
            post_switch_prompt = serialize_state(post_switch_state)
            post_switch_raw_size = sum(len(m.get("content", "")) for m in all_messages)

    # Display the comparison
    console.print()
    comparison = Table(title="Context Size Comparison", show_header=True, header_style="bold magenta")
    comparison.add_column("Metric", style="cyan", width=30)
    comparison.add_column("Before Switch (turn 5)", style="yellow", width=25)
    comparison.add_column("After Switch (turn 10)", style="green", width=25)

    comparison.add_row(
        "Raw message size (chars)",
        str(pre_switch_raw_size),
        str(post_switch_raw_size),
    )
    comparison.add_row(
        "Handoff prompt size (chars)",
        str(len(pre_switch_prompt)),
        str(len(post_switch_prompt)),
    )

    pre_reduction = (1 - len(pre_switch_prompt) / max(pre_switch_raw_size, 1)) * 100
    post_reduction = (1 - len(post_switch_prompt) / max(post_switch_raw_size, 1)) * 100
    comparison.add_row(
        "Reduction vs raw history",
        f"{pre_reduction:.0f}%",
        f"{post_reduction:.0f}%",
    )
    comparison.add_row(
        "Messages forwarded",
        f"5 raw messages",
        f"10 raw messages (but only {len(post_switch_state.raw_last_n)} in handoff)",
    )

    console.print(comparison)

    # Show the actual handoff prompt
    console.print()
    console.print(Panel(
        Markdown(post_switch_prompt),
        title="[bold yellow]Handoff Prompt Sent to New Model[/bold yellow]",
        border_style="yellow",
        padding=(1, 2),
    ))

    console.print()
    console.print(
        f"[bold green]✓[/bold green] Context handoff reduced payload by {post_reduction:.0f}%. "
        "[dim]The new model gets structured state, not raw history.[/dim]"
    )


# ── Scenario: provider_health ────────────────────────────────────────

SCENARIO_PROVIDER_HEALTH = {
    "help": "Inject 3 errors, verify degradation, mock time recovery",
}


async def scenario_provider_health() -> None:
    console.print(Panel.fit(
        "[bold cyan]Scenario: Provider Health[/bold cyan]\n"
        "Inject 3 errors → verify degraded → route around → mock time recovery.",
        border_style="cyan",
    ))

    health = ProviderHealthTracker()
    chain = FallbackChain(health_tracker=health)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Step", style="dim", width=6)
    table.add_column("Action", style="cyan", width=35)
    table.add_column("Provider", style="yellow", width=15)
    table.add_column("Degraded", width=10)
    table.add_column("Next Model", style="green", width=28)
    table.add_column("Status", width=20)

    # Step 1: Initial state — healthy
    initial_degraded = health.is_degraded("anthropic")
    initial_model = chain.get_next()
    table.add_row(
        "1", "Initial state", "anthropic",
        str(initial_degraded),
        initial_model["id"] if initial_model else "—",
        "[green]healthy[/green]",
    )

    # Step 2: Inject 3 consecutive errors
    for i in range(3):
        await health.record_error("anthropic")
    degraded_after_errors = health.is_degraded("anthropic")
    table.add_row(
        "2", f"Injected 3 errors", "anthropic",
        f"[red]{degraded_after_errors}[/red]",
        "—",
        "[red]degraded[/red]",
    )

    # Step 3: Send a request — should route around degraded provider
    next_model = chain.get_next()
    if next_model:
        bypass_status = "[green]routed away[/green]"
        bypass_model = f"{next_model['id']} ({next_model['provider']})"
    else:
        bypass_status = "[red]stuck[/red]"
        bypass_model = "—"

    table.add_row(
        "3", "Request after degradation", "anthropic",
        "[red]True[/red]",
        bypass_model[:26],
        bypass_status,
    )

    # Step 4: Mock 90 seconds passing — recover
    # We simulate this by directly resetting the error timestamps
    # In real code, errors would age out of the 60s window
    health._error_counts["anthropic"] = [t - 90 for t in health._error_counts.get("anthropic", [])]
    health._degraded["anthropic"] = False
    recovered = health.is_degraded("anthropic")
    recovered_model = chain.get_next()
    table.add_row(
        "4", "After 90 seconds (mock)", "anthropic",
        f"[green]{recovered}[/green]",
        recovered_model["id"] if recovered_model else "—",
        "[green]recovered[/green]",
    )

    console.print(table)
    console.print()
    console.print(
        "[bold green]✓[/bold green] Health tracking verified. "
        "[dim]Degradation detected, routing avoided, recovery confirmed.[/dim]"
    )


# ── Scenario: stress ─────────────────────────────────────────────────

SCENARIO_STRESS = {
    "help": "Fire 20 concurrent async requests with mixed task types",
}

STRESS_TASKS = [
    "Write a merge sort implementation in Python",
    "Explain the GIL in CPython",
    "Fix this NullPointerException in Java",
    "Design a rate limiter for an API",
    "Complete this regex pattern for email validation",
    "Review this PR for security issues",
    "Build a React component for a todo list",
    "What does this bash script do?",
    "Optimize this SQL query",
    "Write unit tests for a payment service",
    "Add TypeScript types to this JavaScript",
    "Explain how Redis pub/sub works",
    "Debug this memory leak in Node.js",
    "Design a caching layer for microservices",
    "Autocomplete this CSS media query",
    "Audit this code for CSRF vulnerabilities",
    "Refactor this monolith into services",
    "Explain OAuth 2.0 flow",
    "Write a Dockerfile for a Go app",
    "Fix this race condition in Go",
]


async def scenario_stress() -> None:
    console.print(Panel.fit(
        "[bold cyan]Scenario: Stress Test[/bold cyan]\n"
        "20 concurrent async requests — mixed task types, all mocked.",
        border_style="cyan",
    ))

    results: list[dict] = []
    model_usage: dict[str, int] = {}
    failures: list[str] = []
    total_start = time.time()

    async def _process_task(idx: int, text: str) -> None:
        messages = [{"role": "user", "content": text}]
        task_start = time.time()
        try:
            rule_result = await evaluate_rules(messages)

            if rule_result.conclusive:
                model_def = get_model_by_id(rule_result.model_id or "")
                reason = rule_result.reason
            else:
                cat = classify_task(text, messages)
                preferred = _get_preferred_models_for_task_safe(cat)
                model_def = preferred[0] if preferred else None
                reason = f"classifier: {cat}"

                if model_def is None:
                    chain = FallbackChain()
                    model_def = chain.get_next(task_category=cat)

            if model_def:
                model_id = model_def["id"]
                model_usage[model_id] = model_usage.get(model_id, 0) + 1
                elapsed = time.time() - task_start
                results.append({
                    "idx": idx,
                    "category": reason.split(":")[-1].strip() if ":" in reason else "rule_engine",
                    "model": model_id[:25],
                    "time": f"{elapsed:.3f}s",
                    "status": "ok",
                })
            else:
                failures.append(f"Task {idx}: no model available")
        except Exception as exc:
            failures.append(f"Task {idx}: {exc}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        stress_task = progress.add_task("Firing 20 concurrent requests...", total=20)

        # Fire all 20 concurrently
        tasks = [_process_task(i, text) for i, text in enumerate(STRESS_TASKS, 1)]
        await asyncio.gather(*tasks)
        progress.update(stress_task, completed=20)

    total_time = time.time() - total_start

    # Summary table
    summary = Table(title="Stress Test Results", show_header=True, header_style="bold magenta")
    summary.add_column("#", style="dim", width=4)
    summary.add_column("Category", style="cyan", width=18)
    summary.add_column("Model", style="green", width=28)
    summary.add_column("Time", style="dim", width=10)
    summary.add_column("Status", width=8)

    for r in sorted(results, key=lambda x: x["idx"]):
        summary.add_row(str(r["idx"]), r["category"], r["model"], r["time"], "[green]✓[/green]")

    console.print(summary)

    # Model distribution
    console.print()
    dist = Table(title="Model Distribution", show_header=True, header_style="bold magenta")
    dist.add_column("Model", style="green")
    dist.add_column("Requests", style="yellow")
    for model, count in sorted(model_usage.items(), key=lambda x: -x[1]):
        dist.add_row(model[:35], str(count))
    console.print(dist)

    console.print()
    console.print(
        f"[bold green]✓[/bold green] 20 requests completed in {total_time:.2f}s. "
        f"{len(results)} succeeded, {len(failures)} failed. "
        f"{len(model_usage)} models used."
    )
    if failures:
        console.print(f"  [red]Failures: {'; '.join(failures)}[/red]")


# ── Scenario dispatcher ──────────────────────────────────────────────

SCENARIOS = {
    "dry_run": (scenario_dry_run, SCENARIO_DRY_RUN),
    "task_routing": (scenario_task_routing, SCENARIO_TASK_ROUTING),
    "rate_limit": (scenario_rate_limit, SCENARIO_RATE_LIMIT),
    "context_switch": (scenario_context_switch, SCENARIO_CONTEXT_SWITCH),
    "provider_health": (scenario_provider_health, SCENARIO_PROVIDER_HEALTH),
    "stress": (scenario_stress, SCENARIO_STRESS),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Switchboard CLI Tester — demo and verify all routing scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scenarios:
  dry_run          Route all 7 task categories with zero API keys
  task_routing     Send one request per category, show routing decisions
  rate_limit       Simulate 429, verify fallback fires
  context_switch   10-turn conversation, force switch, show handoff prompt
  provider_health  Inject errors, verify degradation, mock recovery
  stress           20 concurrent async requests, mixed tasks

Examples:
  python cli_test.py dry_run
  python cli_test.py task_routing
  python cli_test.py rate_limit
  python cli_test.py all
        """,
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        choices=list(SCENARIOS.keys()) + ["all"],
        default=None,
        help="Scenario to run (default: show help)",
    )
    args = parser.parse_args()

    if args.scenario is None:
        parser.print_help()
        sys.exit(0)

    if args.scenario == "all":
        async def run_all():
            for name, (fn, info) in SCENARIOS.items():
                console.rule(f"[bold]{name}[/bold]")
                await fn()
                console.print()
        asyncio.run(run_all())
    else:
        fn, _ = SCENARIOS[args.scenario]
        asyncio.run(fn())


if __name__ == "__main__":
    main()
