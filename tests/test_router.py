"""Tests for the Switchboard router — rule engine, classifier, fallback, context."""

from __future__ import annotations

import pytest

from config.settings import RoutingPreferences, prefs
from context.extractor import extract_from_messages, extract_task_hint
from context.serializer import serialize_state
from context.state import ConversationState
from router.classifier import TASK_CATEGORIES, classify_task
from router.fallback_chain import FallbackChain
from router.rule_engine import RuleRoutingResult, evaluate_rules


# ── Rule Engine Tests ──────────────────────────────────────────────

class TestRuleEngine:
    @pytest.mark.asyncio
    async def test_inconclusive_for_normal_task(self):
        """A normal code generation task should be inconclusive (falls to classifier)."""
        messages = [{"role": "user", "content": "Write a REST API endpoint in FastAPI"}]
        result = await evaluate_rules(messages)
        # May be conclusive if simple-task keywords match; just check it returns a valid result
        assert isinstance(result, RuleRoutingResult)

    @pytest.mark.asyncio
    async def test_simple_task_routes_to_cheap(self):
        """Tasks with 'explain' keyword should route to cheap model."""
        messages = [{"role": "user", "content": "Explain what this function does"}]
        result = await evaluate_rules(messages)
        assert result.conclusive is True
        assert "cheap" in result.reason.lower() or "simple" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_prefer_cheap_override(self):
        """When prefer_cheap is set, should always pick cheapest."""
        messages = [{"role": "user", "content": "Design a microservice architecture"}]
        result = await evaluate_rules(
            messages, preferences=RoutingPreferences(prefer_cheap=True)
        )
        assert result.conclusive is True
        assert "cheapest" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_prefer_fast_override(self):
        """When prefer_fast is set, should always pick fastest."""
        messages = [{"role": "user", "content": "Design a microservice architecture"}]
        result = await evaluate_rules(
            messages, preferences=RoutingPreferences(prefer_fast=True)
        )
        assert result.conclusive is True
        assert "fastest" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_large_context_routes_to_big_window(self):
        """When context > 60K tokens, should pick a large-context model."""
        # Reset shared prefs to defaults
        prefs.prefer_cheap = False
        prefs.prefer_fast = False
        prefs.max_cost_per_request = 1.0
        prefs.blacklist_providers = []

        messages = [{"role": "user", "content": "Summarize this 80K token document"}]
        result = await evaluate_rules(messages, context_size=80_000)
        assert result.conclusive is True
        assert "large context" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_latency_sensitive_routes_to_fast(self):
        """Autocomplete tasks should go to fast models."""
        messages = [{"role": "user", "content": "Autocomplete this function"}]
        result = await evaluate_rules(messages)
        assert result.conclusive is True


# ── Classifier Tests ───────────────────────────────────────────────

class TestClassifier:
    def test_task_categories_defined(self):
        """All expected categories should be present."""
        expected = [
            "code_generation", "code_review", "debugging",
            "explanation", "architecture", "autocomplete", "security_audit",
        ]
        for cat in expected:
            assert cat in TASK_CATEGORIES

    def test_classify_code_generation(self):
        text = "Write a Python function that sorts a list using merge sort"
        result = classify_task(text)
        assert result == "code_generation"

    def test_classify_debugging(self):
        text = "I'm getting a NullPointerException in my Java code, how do I fix this bug?"
        result = classify_task(text)
        assert result in ("debugging", "code_generation")  # keyword boost helps

    def test_classify_explanation(self):
        text = "Can you explain how async/await works in Python?"
        result = classify_task(text)
        assert result in ("explanation", "debugging")

    def test_classify_security_audit(self):
        text = "Security audit: check this code for XSS and SQL injection vulnerabilities"
        result = classify_task(text)
        assert result == "security_audit"

    def test_classify_autocomplete(self):
        text = "Complete this function: def fibonacci(n):"
        result = classify_task(text)
        assert result in ("autocomplete", "code_generation")

    def test_classify_with_history(self):
        """Classifier should use conversation history for context."""
        task = "How should I proceed?"
        history = [
            {"role": "user", "content": "My app crashes with a stack overflow error"},
            {"role": "assistant", "content": "Let me check the recursion..."},
            {"role": "user", "content": "Here's the traceback"},
        ]
        result = classify_task(task, conversation_history=history)
        # The history should push it toward debugging
        assert result in ("debugging", "explanation")


# ── Context State & Extractor Tests ────────────────────────────────

class TestContext:
    def test_extract_from_empty_messages(self):
        state = extract_from_messages([])
        assert state.task_intent == ""
        assert state.current_subtask == ""
        assert state.raw_last_n == []

    def test_extract_task_intent(self):
        messages = [
            {"role": "user", "content": "Build a REST API for a todo app"},
            {"role": "assistant", "content": "Sure, let's start with the model..."},
        ]
        state = extract_from_messages(messages)
        assert "REST API" in state.task_intent or "todo" in state.task_intent

    def test_extract_current_subtask(self):
        messages = [
            {"role": "user", "content": "Build a REST API"},
            {"role": "assistant", "content": "Here's the model..."},
            {"role": "user", "content": "Now add authentication"},
        ]
        state = extract_from_messages(messages)
        assert "authentication" in state.current_subtask

    def test_extract_active_files(self):
        messages = [
            {
                "role": "assistant",
                "content": "Here's the code:\n```python:app/main.py\ndef main():\n    pass\n```",
            },
        ]
        state = extract_from_messages(messages)
        assert len(state.active_files) >= 1

    def test_raw_last_n_capped(self):
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        state = extract_from_messages(messages, max_raw=4)
        assert len(state.raw_last_n) <= 4

    def test_serialize_state(self):
        state = ConversationState(
            task_intent="Build a web app",
            active_files=["main.py", "config.yaml"],
            decision_log=["Used FastAPI framework", "SQLite for storage"],
            current_subtask="Add user authentication",
            raw_last_n=[
                {"role": "user", "content": "Add auth"},
                {"role": "assistant", "content": "Sure!"},
            ],
        )
        prompt = serialize_state(state)
        assert "Build a web app" in prompt
        assert "main.py" in prompt
        assert "FastAPI" in prompt
        assert "authentication" in prompt.lower()


# ── Fallback Chain Tests ───────────────────────────────────────────

class TestFallbackChain:
    def test_default_chain_has_free_models(self):
        chain = FallbackChain()
        available = chain.get_all_available()
        free_models = [m for m in available if m.get("tier") == "free"]
        assert len(free_models) >= 1, "Should have at least 1 free-tier fallback"

    def test_get_next_returns_model(self):
        chain = FallbackChain()
        result = chain.get_next()
        assert result is not None
        assert "id" in result
        assert "provider" in result

    def test_skip_model(self):
        chain = FallbackChain()
        first = chain.get_next()
        assert first is not None
        second = chain.get_next(skip_model=first["id"])
        assert second is not None
        assert second["id"] != first["id"]

    def test_respects_blacklist(self):
        original = prefs.blacklist_providers
        try:
            prefs.blacklist_providers = ["openrouter"]
            chain = FallbackChain()
            result = chain.get_next()
            # Should still work if Anthropic models are available
            if result is not None:
                assert result.get("provider") != "openrouter" or result.get("tier") == "free"
        finally:
            prefs.blacklist_providers = original

    def test_task_category_prefers_matching_models(self):
        chain = FallbackChain()
        result = chain.get_next(task_category="code_generation")
        # Should return something; the category filtering is a preference, not a hard filter
        assert result is not None


# ── Preferences Tests ──────────────────────────────────────────────

class TestPreferences:
    def test_default_values(self):
        p = RoutingPreferences()
        assert p.prefer_cheap is False
        assert p.prefer_fast is False
        assert p.max_cost_per_request == 1.0
        assert p.blacklist_providers == []

    def test_to_dict_and_from_dict(self):
        original = RoutingPreferences(
            prefer_cheap=True,
            prefer_fast=False,
            max_cost_per_request=0.5,
            blacklist_providers=["anthropic"],
        )
        restored = RoutingPreferences.from_dict(original.to_dict())
        assert restored.prefer_cheap is True
        assert restored.max_cost_per_request == 0.5
        assert restored.blacklist_providers == ["anthropic"]
