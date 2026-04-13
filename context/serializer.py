"""Serializer — build a compact handoff prompt from ConversationState."""

from __future__ import annotations

from .state import ConversationState

HANDOFF_TEMPLATE = """\
You are continuing a coding session. Here is the current state:

GOAL: {task_intent}
ACTIVE FILES: {active_files}
CHANGES MADE SO FAR:
{code_diffs}
KEY DECISIONS:
{decision_log}
CURRENT SUBTASK: {current_subtask}
Recent conversation:
{raw_last_n}

Continue from here.
"""


def serialize_state(state: ConversationState) -> str:
    """Convert a :class:`ConversationState` into a handoff prompt string."""
    active_files_str = "\n".join(f"  - {f}" for f in state.active_files) or "  (none)"
    diffs_str = "\n---\n".join(state.code_diffs) or "  (no changes yet)"
    decisions_str = "\n".join(f"  - {d}" for d in state.decision_log) or "  (none recorded)"

    # Format raw messages compactly
    raw_lines: list[str] = []
    for msg in state.raw_last_n:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        raw_lines.append(f"[{role}]: {content[:300]}")
    raw_str = "\n".join(raw_lines) or "  (empty)"

    return HANDOFF_TEMPLATE.format(
        task_intent=state.task_intent or "(not specified)",
        active_files=active_files_str,
        code_diffs=diffs_str,
        decision_log=decisions_str,
        current_subtask=state.current_subtask or "(not specified)",
        raw_last_n=raw_str,
    )


def build_handoff_messages(
    state: ConversationState,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Return an OpenAI-format message list from the serialized state.

    Returns ``[system, user]`` or ``[user]`` depending on whether a
    *system_prompt* is given.
    """
    prompt = serialize_state(state)
    msgs: list[dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": prompt})
    return msgs
