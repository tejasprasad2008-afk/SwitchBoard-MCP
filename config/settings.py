"""Settings and configuration loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# ── Paths ──────────────────────────────────────────────────────────

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_MODELS_YAML = _PACKAGE_ROOT / "config" / "models.yaml"
_STATE_DIR = Path.home() / ".switchboard"
_LOG_FILE = _STATE_DIR / "routing.log"
_DB_FILE = _STATE_DIR / "state.sqlite"

# Create state directory with restricted permissions (0700)
_STATE_DIR.mkdir(parents=True, exist_ok=True)
os.chmod(_STATE_DIR, 0o700)

# ── API Keys ───────────────────────────────────────────────────────


def get_anthropic_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY")


def get_openrouter_key() -> str | None:
    return os.environ.get("OPENROUTER_API_KEY")


def get_openai_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY")


# ── Model Registry ─────────────────────────────────────────────────

_model_registry: list[dict[str, Any]] | None = None


def load_models(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load model definitions from YAML. Cached after first load."""
    global _model_registry
    if _model_registry is not None:
        return _model_registry

    src = Path(path) if path else _MODELS_YAML
    with open(src, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    _model_registry = data.get("models", [])
    return _model_registry


def get_model_by_id(model_id: str) -> dict[str, Any] | None:
    """Look up a model definition by its id."""
    for m in load_models():
        if m["id"] == model_id:
            return m
    return None


def get_models_by_strength(strength: str) -> list[dict[str, Any]]:
    """Return models that list *strength* in their strengths, ordered by cost."""
    matches = [m for m in load_models() if strength in m.get("strengths", [])]
    return sorted(matches, key=lambda m: m.get("cost_per_1k_tokens", 0))


# ── Runtime Preferences ────────────────────────────────────────────

class RoutingPreferences:
    """Mutable runtime preferences for routing behaviour."""

    def __init__(
        self,
        prefer_cheap: bool = False,
        prefer_fast: bool = False,
        max_cost_per_request: float = 1.0,
        blacklist_providers: list[str] | None = None,
    ) -> None:
        self.prefer_cheap = prefer_cheap
        self.prefer_fast = prefer_fast
        self.max_cost_per_request = max_cost_per_request
        self.blacklist_providers = blacklist_providers or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefer_cheap": self.prefer_cheap,
            "prefer_fast": self.prefer_fast,
            "max_cost_per_request": self.max_cost_per_request,
            "blacklist_providers": self.blacklist_providers,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoutingPreferences":
        return cls(
            prefer_cheap=data.get("prefer_cheap", False),
            prefer_fast=data.get("prefer_fast", False),
            max_cost_per_request=data.get("max_cost_per_request", 1.0),
            blacklist_providers=data.get("blacklist_providers", []),
        )


# ── Global preferences instance ────────────────────────────────────

prefs = RoutingPreferences()
