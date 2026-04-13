"""Extractor — pull intent, files, diffs, and decisions from raw messages."""

from __future__ import annotations

import re
from typing import Any

from .state import ConversationState

# Patterns for extracting structured info from assistant messages
_DIFF_RE = re.compile(r"```(?:diff|diff\s+.*?)\n(.*?)```", re.DOTALL)
_FILE_RE = re.compile(r"```(?:\w+:)?\s*([^\s`/]+(?:/[^\s`]+)*\.\w+)")
_DECISION_RE = re.compile(
    r"(?:decided|decide|chose|use|using|adopted)\s+(?:to\s+)?(?:use|the|a|an)\s+(.*?)(?:\.|\n)",
    re.IGNORECASE,
)


def extract_from_messages(
    messages: list[dict[str, str]],
    max_raw: int = 4,
) -> ConversationState:
    """Build a :class:`ConversationState` from a raw message list.

    Heuristics:
    - The **last user message** is taken as the current sub-task hint.
    - Assistant messages are scanned for code blocks (file paths), diffs,
      and decision-like sentences.
    - The **first user message** (if any) is treated as the overarching task intent.
    """
    state = ConversationState()

    if not messages:
        return state

    # ── Task intent from first user message ────────────────────────
    for msg in messages:
        if msg.get("role") == "user":
            state.task_intent = msg.get("content", "")[:500]
            break

    # ── Current subtask from last user message ─────────────────────
    for msg in reversed(messages):
        if msg.get("role") == "user":
            state.current_subtask = msg.get("content", "")[:500]
            break

    # ── Scan assistant messages for structured info ────────────────
    seen_files: set[str] = set()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")

        # File paths from code fences
        for m in _FILE_RE.finditer(content):
            path = m.group(1)
            if path not in seen_files:
                seen_files.add(path)
                state.active_files.append(path)

        # Diff blocks
        for m in _DIFF_RE.finditer(content):
            diff_text = m.group(1).strip()
            if diff_text:
                state.code_diffs.append(diff_text[:1000])

        # Decision sentences
        for m in _DECISION_RE.finditer(content):
            decision = m.group(1).strip()
            if len(decision) > 5 and decision not in state.decision_log:
                state.decision_log.append(decision[:200])

    # ── Last N raw messages (keep both roles for continuity) ───────
    state.raw_last_n = messages[-max_raw:]

    return state


def extract_task_hint(messages: list[dict[str, str]]) -> str:
    """Return a short string describing the apparent task."""
    # Use the last user message as the primary hint
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")[:300]
    return ""


def extract_file_context(messages: list[dict[str, str]]) -> str:
    """Return active file paths as a comma-separated string."""
    state = extract_from_messages(messages, max_raw=0)
    return ", ".join(state.active_files) if state.active_files else ""
