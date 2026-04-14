"""ConversationState dataclass — the structured context that survives model switches."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationState:
    """Structured representation of a coding session that can be serialized
    into a compact handoff prompt when switching models."""

    task_intent: str = ""                # what the user is ultimately trying to do
    active_files: list[str] = field(default_factory=list)  # files being edited
    code_diffs: list[str] = field(default_factory=list)    # changes made (git diff format)
    decision_log: list[str] = field(default_factory=list)  # key design decisions
    current_subtask: str = ""            # what the current turn is trying to accomplish
    raw_last_n: list[dict] = field(default_factory=list)   # last N raw messages

    def to_dict(self) -> dict:
        return {
            "task_intent": self.task_intent,
            "active_files": self.active_files,
            "code_diffs": self.code_diffs,
            "decision_log": self.decision_log,
            "current_subtask": self.current_subtask,
            "raw_last_n": self.raw_last_n,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConversationState:
        return cls(
            task_intent=data.get("task_intent", ""),
            active_files=data.get("active_files", []),
            code_diffs=data.get("code_diffs", []),
            decision_log=data.get("decision_log", []),
            current_subtask=data.get("current_subtask", ""),
            raw_last_n=data.get("raw_last_n", []),
        )
