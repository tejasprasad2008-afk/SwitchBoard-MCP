# Contributing to Switchboard

Thanks for wanting to help. This project is built for individual developers who just want their coding agent to keep working — no enterprise infra, no vendor lock-in. Every contribution should serve that goal.

---

## Dev Setup

```bash
# 1. Clone and set up venv
git clone https://github.com/yourname/switchboard-mcp.git
cd switchboard-mcp
python3 -m venv .venv && source .venv/bin/activate

# 2. Install with dev dependencies
pip install -e ".[dev]"

# 3. Run all tests
pytest tests/ -v

# 4. Run the CLI tester (zero API keys needed)
python cli_test.py dry_run
```

---

## Adding a New Provider

Providers are how Switchboard talks to different AI APIs. There are currently two: Anthropic (direct) and OpenRouter (gateway to 12+ models).

### Steps

**1. Create the provider file**

Create `providers/your_provider.py` extending the base class:

```python
from providers.base import BaseProvider

class YourProvider(BaseProvider):
    name = "your_provider"

    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> dict | AsyncIterator[str]:
        # Return: {"text": str, "model": str, "provider": str, "usage": {...}}
        pass

    async def health_check(self) -> bool:
        # Return True if the provider is reachable
        pass
```

**2. Register in the model registry**

Add entries to `config/models.yaml`:

```yaml
- id: your-provider/some-model
  provider: your_provider          # Must match BaseProvider.name
  cost_per_1k_tokens: 0.001
  context_window: 128000
  strengths: [code_generation, debugging]
  speed: fast
  tier: paid
```

**3. Add to the server's provider cache**

In `server.py`, add the import and registration in `_get_provider()`:

```python
from providers.your_provider import YourProvider

def _get_provider(provider_name: str):
    if provider_name not in _provider_cache:
        # ... existing providers ...
        elif provider_name == "your_provider":
            _provider_cache[provider_name] = YourProvider()
```

**4. Write tests**

Add tests to `tests/test_integration.py`:

```python
class TestYourProvider:
    @pytest.mark.asyncio
    async def test_chat_complete(self):
        with respx.mock:
            respx.post("https://api.your-provider.com/...").mock(
                return_value=Response(200, json={"text": "hello"})
            )
            provider = YourProvider(api_key="test")
            result = await provider.chat_complete(
                messages=[{"role": "user", "content": "hi"}],
                model="some-model",
            )
            assert result["text"] == "hello"
```

---

## Adding a New Model

You don't need to touch any Python code. Just edit `config/models.yaml`:

```yaml
- id: provider/model-name          # Must match what the API expects
  provider: openrouter             # or 'anthropic', or your custom provider
  cost_per_1k_tokens: 0.001        # in USD
  context_window: 128000           # in tokens
  strengths: [code_generation, debugging]  # from the 7 categories
  speed: fast                      # 'fast', 'medium', or 'slow'
  tier: paid                       # 'paid' or 'free'
```

**Fields explained:**

| Field | Description | Example |
|---|---|---|
| `id` | The model identifier sent to the API | `anthropic/claude-sonnet-4` |
| `provider` | Which provider handles this model | `openrouter`, `anthropic` |
| `cost_per_1k_tokens` | Cost in USD per 1K input+output tokens | `0.003` |
| `context_window` | Maximum context window in tokens | `200000` |
| `strengths` | Task categories this model excels at | `[code_generation, debugging]` |
| `speed` | Relative inference speed | `fast`, `medium`, `slow` |
| `tier` | Whether this is free or paid | `free`, `paid` |

After adding the model, it's immediately available. No restart needed.

---

## Improving the Classifier

The classifier uses a hybrid approach:

1. **Semantic score** (60%): Cosine similarity between task text and category name embeddings via `all-MiniLM-L6-v2`
2. **Keyword score** (40%): Fraction of category-specific keywords found in the task text

### To improve classification accuracy

**Add keywords** to `router/classifier.py`:

```python
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "code_generation": [
        "write", "create", "implement", "build",
        # ... add more keywords here
    ],
    # ...
}
```

**Test stability:**

```bash
pytest tests/test_router.py::TestClassifierStability -v
```

Each test runs the classifier 50 times on the same input and verifies the result is consistent.

**Why hybrid?** Pure semantic similarity can be fooled by synonyms and phrasing. Pure keyword matching misses context. Together they're robust — the semantic score catches intent, the keyword score provides domain-specific precision.

---

## PR Checklist

Before submitting a PR:

- [ ] All tests pass: `pytest tests/ -v`
- [ ] New models added to `config/models.yaml` (if applicable)
- [ ] CHANGELOG.md updated with your change
- [ ] No TODOs, stubs, or placeholder text in new code
- [ ] Code is typed (use `mypy` if available: `mypy switchboard/`)
- [ ] Linting passes: `ruff check .`
- [ ] The `dry_run` CLI scenario still works: `python cli_test.py dry_run`

---

## Issue Templates

We have three issue templates:

1. **Bug Report** — Something isn't working. Please include your MCP client, provider that failed, and a routing log snippet.
2. **Model Request** — Want a new model added? Just provide the OpenRouter ID (or API details) and we'll add it.
3. **Feature Request** — General improvements and new features.

Use the right template — it helps us triage faster.

---

## Code Style

- **Types everywhere** — function signatures must be typed
- **Async by default** — all I/O uses `async`/`await` with `httpx`
- **No blocking** — never block the MCP event loop
- **Error handling** — every provider call has error handling and fallback logic
- **No secrets** — API keys only come from environment variables, never hardcoded

---

## How Decisions Work

This is a consensus-driven project. Major changes need at least one other maintainer to agree. Small fixes (typos, docs, bug fixes) can be merged directly.

The goal is to move fast without breaking things — every change should make the routing smarter, more reliable, or cheaper for the developer.
