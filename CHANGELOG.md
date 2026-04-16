# Changelog

All notable changes to Switchboard will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-13

### Added

- **MCP Server** with 4 tools: `route_completion`, `get_routing_status`, `set_routing_preferences`, `report_outcome`
- **Hybrid routing engine**: Layer 1 rule engine (fast signals) + Layer 2 sentence-transformer classifier (7 task categories)
- **Context serialization**: Structured `ConversationState` with handoff prompts instead of raw message forwarding — 60-90% context reduction
- **14-model registry** across 3 providers (Anthropic direct, OpenRouter gateway) and 3 tiers (paid, cheap, free)
- **Health-aware fallback chain**: Per-provider error tracking, rate limit detection, degradation marking, SQLite persistence
- **Free-tier safety net**: 2 free models (LLaMA 3.3 70B, Qwen 2.5 72B) as final fallback — agent never hard-stops
- **Feedback loop**: `report_outcome` tool for quality ratings that update provider health scores
- **Dry-run mode**: `--dry-run` flag and `cli_test.py dry_run` scenario for zero-setup demo (zero API keys needed)
- **CLI tester**: 6 scenarios with Rich-formatted output (`rate_limit`, `task_routing`, `context_switch`, `provider_health`, `dry_run`, `stress`)
- **JSONL routing log**: All routing decisions logged to `~/.switchboard/routing.log` for transparency
- **26 unit tests** + **18 integration tests** = **44 total tests**, all passing
- **GitHub Actions CI**: Python 3.10/3.11/3.12 matrix, pytest, codecov
- **MIT License** — built for individual developers
