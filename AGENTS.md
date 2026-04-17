# AGENTS.md

## Dev Setup

```bash
cd switchboard
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Commands

- **Tests:** `pytest tests/ -v`
- **Linting:** `ruff check .`
- **Single test:** `pytest tests/test_router.py::TestClassifierStability -v`
- **CLI tester (no API keys):** `python cli_test.py dry_run`

## Architecture

- **Entry point:** `server:main` (defined in `pyproject.toml`)
- **Package root:** `switchboard/`
- **Models:** Edit `config/models.yaml` — no code changes needed to add models
- **Providers:** `providers/` — extend `providers/base.py`
- **Config:** `config/settings.py` loads from `config/models.yaml`

## Key Files

| File | Purpose |
|------|---------|
| `server.py` | MCP server, provider registration |
| `providers/anthropic.py` | Anthropic direct API |
| `providers/openrouter.py` | OpenRouter gateway |
| `providers/base.py` | Base provider interface |
| `config/models.yaml` | Model registry (14 models) |
| `context/` | Conversation state serialization |

## Gotchas

- **Environment variables:** `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` (not hardcoded)
- **Async only:** All I/O uses `async`/`await` with `httpx`
- **Test mode:** Use `pytest` with `asyncio_mode = "auto"` (in `pyproject.toml`)
- **Provider registration:** Add new providers in `server.py` `_get_provider()` function

## Order of Operations

After code changes: `ruff check .` → `pytest tests/ -v`
