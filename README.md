# Switchboard 🎛️

> Never let your AI coding agent hit a dead end again.

[![PyPI version](https://img.shields.io/pypi/v/switchboard-mcp.svg)](https://pypi.org/project/switchboard-mcp/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/yourname/switchboard-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/yourname/switchboard-mcp/actions)
[![CLI Demo](https://img.shields.io/badge/CLI-zero--setup--demo-orange)](#cli-tester)

A multi-provider AI routing layer that gives any MCP-compatible coding agent (Claude Code, Cursor, Windsurf) **unlimited flow** by transparently switching between AI providers when rate limits hit or when a better model exists for the task.

---

## The Problem

It's 2am. You're deep in a coding session with Claude Code. You've got 80 messages of context, the agent is about to write the perfect solution — and then:

> *"You've reached your message limit. Please upgrade your plan or wait."*

The agent stops dead. Your flow is gone. You're staring at an error message instead of working code.

**This shouldn't happen.** Your coding agent should never hit a wall because one provider said no.

Switchboard fixes this. It sits between your coding agent and the AI providers, watching for rate limits, picking the best model for each task, and silently falling back to alternatives — all without the agent (or you) ever knowing a switch happened.

---

## How It Works

```
┌──────────────────────────────────────────────────────────────────┐
│                    YOUR WORKSTATION                               │
│                                                                  │
│  ┌─────────────┐                                                 │
│  │ Your Agent  │   Claude Code / Cursor / Windsurf               │
│  │ (Claude etc)│                                                 │
│  └──────┬──────┘                                                 │
│         │  MCP tool calls only                                    │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────┐        │
│  │           SWITCHBOARD (The Whisperer)                 │        │
│  │                                                      │        │
│  │  ┌──────────┐   ┌───────────┐   ┌──────────────┐    │        │
│  │  │ Layer 1  │──▶│  Layer 2  │──▶│   Fallback   │    │        │
│  │  │ Rules    │   │ Classifier│   │   Chain      │    │        │
│  │  │ (fast)   │   │ (DistilBERT)│  │ + Health     │    │        │
│  │  └──────────┘   └───────────┘   └──────┬───────┘    │        │
│  │                                        │             │        │
│  │  ┌──────────────────────────────────┐  │             │        │
│  │  │ Context Serializer (Handoff)     │──┘             │        │
│  │  │ Structured state ≠ raw history   │                │        │
│  │  └──────────────────────────────────┘                │        │
│  └───────────────────────┬──────────────────────────────┘        │
│                          │                                       │
│         ┌────────────────┼────────────────┐                     │
│         ▼                ▼                ▼                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│  │ Anthropic│    │OpenRouter│    │ Direct   │                  │
│  │ (Claude) │    │(14 models│    │ APIs     │                  │
│  └──────────┘    └──────────┘    └──────────┘                  │
└──────────────────────────────────────────────────────────────────┘
```

### Routing Decision Flow

```
Request arrives at Switchboard
       │
       ▼
┌──────────────────┐
│  Layer 1: Rules  │  ← Checks in order:
│  (instant)       │     1. User preferences (cheap/fast)?
└────────┬─────────┘     2. Context > 60K tokens?
         │                3. Simple task (explain, autocomplete)?
    conclusive?           4. Latency-sensitive?
         │                5. Budget cap?
    ┌────┴────┐
    │         │
   YES        NO
    │         │
    ▼         ▼
 Route to   ┌──────────────────┐
 chosen     │ Layer 2:         │  ← Hybrid classifier:
 model      │ Classifier       │     60% semantic (sentence-transformers)
            │ (all-MiniLM-L6)  │     40% keyword overlap
            └────────┬─────────┘
                     │
                     ▼
              Best model for task category
                     │
                     ▼
            ┌──────────────────┐
            │  Fallback Chain  │  ← If model fails:
            │  (health-aware)  │     1. Try next in priority list
            └────────┬─────────┘     2. Skip degraded/rate-limited
                     │               3. Fall to free-tier models
                     ▼               4. Never hard-stop
              Response back to agent
              (switch is invisible)
```

---

## Quick Start

### Install

```bash
pip install switchboard-mcp
```

### Set your API keys

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENROUTER_API_KEY="sk-or-..."
```

### Add to your MCP client

Every MCP client speaks the same stdio transport protocol. Your server runs as a subprocess, the client connects, discovers the 4 tools, and starts calling them. Switchboard is completely client-agnostic.

The core config block is always the same — just the root key changes per client:

```json
{
  "command": "switchboard",
  "env": {
    "ANTHROPIC_API_KEY": "sk-ant-...",
    "OPENROUTER_API_KEY": "sk-or-..."
  }
}
```

Here's exactly where it goes for each client. **Find yours, copy, paste.**

---

#### Client Configuration Reference

| Client | Config File / Location | Root Key | Notes |
|---|---|---|---|
| [Cursor](#cursor) | `~/.cursor/mcp.json` | `"mcpServers"` | Or via GUI: Settings → Features → MCP |
| [VS Code (Copilot)](#vs-code-copilot) | `.vscode/mcp.json` | `"servers"` | Workspace-level |
| [Claude Code](#claude-code) | `~/.claude/settings.json` | `"mcpServers"` | Global |
| [Windsurf](#windsurf) | Settings → MCP Servers | `"mcpServers"` | Same JSON structure as Cursor |
| [JetBrains](#jetbrains) | Settings → AI Assistant → MCP | `"mcpServers"` | GUI paste or auto-configure |
| [Antigravity / Google ADK](#antigravity--google-adk) | CLI config | stdio | One config, multiple agents |
| [GitKraken](#gitkraken) | MCP settings | `"mcpServers"` | Same as Cursor |

---

##### Cursor

**File:** `~/.cursor/mcp.json` (or Settings → Features → MCP → Add Server)

```json
{
  "mcpServers": {
    "switchboard": {
      "command": "switchboard",
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "OPENROUTER_API_KEY": "sk-or-..."
      }
    }
  }
}
```

Restart Cursor. Your agent now has unlimited flow.

##### VS Code (Copilot)

**File:** `.vscode/mcp.json` (workspace-level)

> **Note:** VS Code uses `"servers"` as the root key, not `"mcpServers"`.

```json
{
  "servers": {
    "switchboard": {
      "command": "switchboard",
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "OPENROUTER_API_KEY": "sk-or-..."
      }
    }
  }
}
```

Reload the window (`Ctrl+Shift+P` → "Developer: Reload Window").

##### Claude Code

**File:** `~/.claude/settings.json`

```json
{
  "mcpServers": {
    "switchboard": {
      "command": "switchboard",
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "OPENROUTER_API_KEY": "sk-or-..."
      }
    }
  }
}
```

Next `claude` invocation picks it up automatically.

##### Windsurf

**Location:** Settings → MCP Servers → Add Server

Use the same JSON structure as Cursor:

```json
{
  "mcpServers": {
    "switchboard": {
      "command": "switchboard",
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "OPENROUTER_API_KEY": "sk-or-..."
      }
    }
  }
}
```

##### JetBrains

**Location:** Settings → Tools → AI Assistant → Model Context Protocol (MCP) → Add → "As JSON ChatForest"

Paste this config:

```json
{
  "mcpServers": {
    "switchboard": {
      "command": "switchboard",
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "OPENROUTER_API_KEY": "sk-or-..."
      }
    }
  }
}
```

> **Tip:** JetBrains can also auto-configure external clients from the IDE side. If you're already using other MCP servers, just add Switchboard alongside them.

##### Antigravity / Google ADK

Uses the same stdio config. Configure once, use across multiple agents:

```json
{
  "mcpServers": {
    "switchboard": {
      "command": "switchboard",
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "OPENROUTER_API_KEY": "sk-or-..."
      }
    }
  }
}
```

##### GitKraken

**Location:** MCP Settings → Add Server

Same JSON structure as Cursor:

```json
{
  "mcpServers": {
    "switchboard": {
      "command": "switchboard",
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "OPENROUTER_API_KEY": "sk-or-..."
      }
    }
  }
}
```

---

### Using a Virtual Environment

If `switchboard` isn't on your `PATH` (you installed from source or in a venv), use the full path:

```json
{
  "mcpServers": {
    "switchboard": {
      "command": "/path/to/your/venv/bin/switchboard",
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "OPENROUTER_API_KEY": "sk-or-..."
      }
    }
  }
}
```

Or with `python -m`:

```json
{
  "mcpServers": {
    "switchboard": {
      "command": "python",
      "args": ["-m", "server"],
      "cwd": "/path/to/switchboard/package",
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "OPENROUTER_API_KEY": "sk-or-..."
      }
    }
  }
}
```

That's it. Your agent now has unlimited flow. When Claude hits a rate limit, Switchboard silently routes to DeepSeek, GPT-4o, or Gemini — your agent never knows.

---

## Model Registry

14 models across 3 tiers, from premium to free fallback:

| Model | Provider | Cost/1K | Context | Best For | Tier |
|---|---|---|---|---|---|
| claude-sonnet-4 | Anthropic | $0.003 | 200K | code gen, review, arch | Paid |
| claude-opus-4 | Anthropic | $0.015 | 200K | review, security, arch | Paid |
| claude-haiku-3.5 | Anthropic | $0.0008 | 200K | explanation, autocomplete | Paid |
| claude-sonnet-4 | OpenRouter | $0.003 | 200K | code gen, review | Paid |
| deepseek-v3 | OpenRouter | $0.00014 | 128K | code gen, debugging | Paid |
| deepseek-r1 | OpenRouter | $0.0005 | 128K | debugging, review | Paid |
| gpt-4o | OpenRouter | $0.0025 | 128K | code gen, review | Paid |
| gpt-4o-mini | OpenRouter | $0.00015 | 128K | autocomplete, explanation | Paid |
| o3-mini | OpenRouter | $0.0011 | 200K | debugging, code gen | Paid |
| gemini-1.5-pro | OpenRouter | $0.00125 | 2M | code gen, architecture | Paid |
| gemini-2.0-flash | OpenRouter | $0.0001 | 1M | explanation, autocomplete | Paid |
| qwen-2.5-coder | OpenRouter | $0.00018 | 128K | code gen, autocomplete | Paid |
| **llama-3.3-70b:free** | OpenRouter | **$0** | 128K | explanation, autocomplete | **Free** |
| **qwen-2.5-72b:free** | OpenRouter | **$0** | 32K | explanation, autocomplete | **Free** |

Full definitions in [`config/models.yaml`](config/models.yaml). Add your own models by editing this file.

---

## Routing Logic

### Hybrid Approach: Rules First, Classifier Second

**Layer 1 (Rule Engine)** runs in < 1ms. It checks:

| Signal | Action |
|---|---|
| `prefer_cheap` is set | Route to cheapest available model |
| `prefer_fast` is set | Route to fastest available model |
| Context > 60K tokens | Route to large-window model (Gemini 1.5 Pro: 2M) |
| Simple task keywords (explain, autocomplete, rename) | Route to cheapest model |
| Latency-sensitive (autocomplete, suggest, stream) | Route to fast model |
| `max_cost_per_request` budget | Filter to affordable models |

**Layer 2 (Classifier)** runs when rules are inconclusive. A sentence-transformer (`all-MiniLM-L6-v2`, 22MB) classifies the task into 7 categories and picks the best model for that task type:

| Category | Preferred Models |
|---|---|
| `code_generation` | Sonnet, DeepSeek-V3, GPT-4o |
| `code_review` | Opus, GPT-5 |
| `debugging` | DeepSeek-R1, o3-mini |
| `explanation` | Haiku, Gemini Flash |
| `architecture` | Opus, GPT-5 |
| `autocomplete` | Haiku, GPT-4o-mini, Qwen-Coder |
| `security_audit` | Opus, GPT-5 |

### Fallback Priority

When a model fails, Switchboard walks down this chain:

1. `claude-sonnet-4` (Anthropic direct)
2. `anthropic/claude-sonnet-4` (OpenRouter)
3. `deepseek/deepseek-v3`
4. `openai/gpt-4o`
5. `google/gemini-1.5-pro`
6. `qwen/qwen-2.5-coder-32b-instruct`
7. `meta-llama/llama-3.3-70b-instruct:free` ← **free tier**
8. `qwen/qwen-2.5-72b-instruct:free` ← **free tier (final safety net)**

**The agent NEVER hard-stops.** Steps 7-8 are always free.

---

### Context Serialization: Why This Isn't LiteLLM

When you switch models mid-conversation, most routers just forward the entire message history. That's wasteful and often breaks things (different models have different context windows and system prompt expectations).

Switchboard maintains a **structured `ConversationState`**:

```
GOAL: Build a REST API for a todo app
ACTIVE FILES:
  - app/main.py
  - app/models.py
CHANGES MADE SO FAR:
  + Added FastAPI setup
  + Created Todo model
KEY DECISIONS:
  - Used SQLite for persistence
CURRENT SUBTASK: Add authentication
Recent conversation:
[user]: Add JWT authentication
[assistant]: I'll add middleware...
```

This handoff prompt is **60-90% smaller** than forwarding raw history, and the new model gets everything it actually needs to continue.

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Recommended | Direct access to Claude models |
| `OPENROUTER_API_KEY` | Recommended | Access to 12+ models via OpenRouter |
| `OPENAI_API_KEY` | Optional | Future direct OpenAI support |

### Runtime Preferences

Set at runtime via `set_routing_preferences`:

```json
{
  "prefer_cheap": true,
  "prefer_fast": false,
  "max_cost_per_request": 0.50,
  "blacklist_providers": ["openrouter"]
}
```

### Customizing Models

Edit `config/models.yaml` to add models or adjust costs:

```yaml
models:
  - id: your/custom-model
    provider: openrouter
    cost_per_1k_tokens: 0.001
    context_window: 128000
    strengths: [code_generation, debugging]
    speed: fast
    tier: paid
```

### Logs

All routing decisions are logged to `~/.switchboard/routing.log` in JSONL format:

```bash
tail -f ~/.switchboard/routing.log | jq .
```

### Dry Run

See what Switchboard would do without making any API calls:

```bash
switchboard --dry-run
```

Or via CLI tester:

```bash
python cli_test.py dry_run
```

---

## CLI Tester

The CLI tester demonstrates every routing scenario with beautiful Rich-formatted output. **No API keys needed** — all HTTP is mocked.

```bash
pip install switchboard-mcp[dev]
python cli_test.py dry_run
```

### Available Scenarios

| Command | What It Shows |
|---|---|
| `python cli_test.py dry_run` | All 7 task categories routed — zero API keys |
| `python cli_test.py task_routing` | Full routing table with classifications |
| `python cli_test.py rate_limit` | 429 handling and fallback |
| `python cli_test.py context_switch` | Structured handoff vs raw history comparison |
| `python cli_test.py provider_health` | Error injection, degradation, recovery |
| `python cli_test.py stress` | 20 concurrent requests, mixed tasks |
| `python cli_test.py all` | Run all scenarios |

This is the zero-setup way to experience Switchboard. Someone finds this repo, runs `python cli_test.py dry_run`, sees the routing decisions in a pretty table — that's the "oh this is real" moment.

---

## Contributing

### Quick Start

```bash
git clone https://github.com/yourname/switchboard-mcp.git
cd switchboard-mcp
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

### Adding a New Provider

1. Create `providers/your_provider.py` extending `providers/base.py`
2. Implement `chat_complete()` and `health_check()`
3. Register in `config/models.yaml` with the correct provider name
4. Add tests to `tests/test_integration.py`

### Adding a New Model

Just edit `config/models.yaml`. No code changes needed:

```yaml
- id: provider/model-name
  provider: openrouter     # or 'anthropic'
  cost_per_1k_tokens: 0.001
  context_window: 128000
  strengths: [code_generation, debugging]
  speed: fast
  tier: paid
```

### Improving the Classifier

The classifier uses hybrid semantic + keyword scoring. To improve it:

1. Add keywords to `CATEGORY_KEYWORDS` in `router/classifier.py`
2. The semantic embeddings will automatically adapt to new text
3. Run `pytest tests/test_router.py::TestClassifierStability` to verify

### PR Checklist

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] New models added to `config/models.yaml`
- [ ] CHANGELOG.md updated
- [ ] No TODOs or stubs left in new code

See [CONTRIBUTING.md](CONTRIBUTING.md) for full details.

---

## Roadmap

| Version | Feature |
|---|---|
| **v0.1.0** | Initial release — hybrid router, 14 models, context handoff, health tracking |
| **v0.2.0** | Web dashboard for routing analytics and cost tracking |
| **v0.3.0** | Fine-tuned classifier trained on real routing data |
| **v0.4.0** | Cost analytics — per-session, per-project cost tracking and budgets |
| **v0.5.0** | Community model marketplace — share routing rules and model configs |

---

## What Makes This Different

| Feature | Switchboard | LiteLLM / OpenRouter Router |
|---|---|---|
| MCP-native | ✅ Zero-code-change for Claude Code, Cursor | ❌ Requires SDK integration |
| Semantic context | ✅ Structured handoff, not raw history | ❌ Raw message forwarding only |
| Task-aware routing | ✅ Understands debug vs review vs gen | ❌ Cost/availability only |
| Feedback loop | ✅ `report_outcome` tool | ❌ None |
| Free-tier fallback | ✅ Always 2+ free models | ⚠️ Depends on plan |
| License | ✅ MIT, for individual devs | ⚠️ Hosted service / Enterprise |

---

## License

[MIT](LICENSE) — built for individual developers, not enterprise.

Made with ☕ at 2am because the rate limit wall needed to go.
