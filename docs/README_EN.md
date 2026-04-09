![Grok Search MCP](../images/hero-banner.svg)

<div align="center">

English | [简体中文](../README.md)

**A standalone search MCP focused on precise search, aggregation, web fetching, and evidence organization.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastMCP 2.0+](https://img.shields.io/badge/FastMCP-2.0%2B-green.svg)](https://github.com/jlowin/fastmcp)
[![MCP Tools 13](https://img.shields.io/badge/MCP%20Tools-13-0ea5e9.svg)](#tools)
[![Clients Codex Claude Gemini](https://img.shields.io/badge/Clients-Codex%20%7C%20Claude%20%7C%20Gemini-111827.svg)](#client-setup)

</div>

---

## What This Project Is

`grok-search` is a [FastMCP](https://github.com/jlowin/fastmcp) server built for one job: **search better**.

It provides:

- `web_search` for AI search plus multi-source aggregation
- `get_sources` for sources, search trace, and evidence bindings
- `web_fetch` for page extraction
- `web_map` for site discovery
- `plan_*` tools for structured search planning

> Note: this repository is a **standalone search MCP**. It is not related to `cccc` or `webcoding`.

## Why This Version

This version focuses on a few concrete improvement areas:

- more robust validation
- explicit error surfacing
- streaming fallback
- persistent source cache
- richer source metadata
- search trace and evidence bindings
- verified client support for `Codex`, `Claude`, and `Gemini`

## Architecture

```mermaid
flowchart LR
    C[Codex / Claude / Gemini / Cherry Studio] --> M[grok-search MCP]

    M --> WS[web_search]
    M --> GS[get_sources]
    M --> WF[web_fetch]
    M --> WM[web_map]
    M --> PL[plan_*]

    WS --> G[Grok API]
    WS --> T[Tavily Search]
    WS --> F[Firecrawl Search]

    WF --> TE[Tavily Extract]
    TE -->|fallback| FS[Firecrawl Scrape]

    WM --> TM[Tavily Map]
    GS --> CACHE[Sources Cache]
    WS --> CACHE
```

## Key Capabilities

| Capability | Current Implementation |
|-----------|------------------------|
| Precise search | time-context injection only when needed, model validation, streaming fallback |
| Aggregation | Grok + Tavily + Firecrawl under a bounded search budget |
| Source auditability | `sources`, `search_trace`, `evidence_bindings` |
| Web fetching | Tavily Extract first, Firecrawl Scrape fallback |
| Site discovery | Tavily Map |
| Complex research | six planning tools from `plan_intent` to `plan_execution` |
| Persistent cache | `get_sources` survives process restarts |
| Multi-client support | validated with `Codex`, `Claude`, `Gemini` |

![](../images/capabilities-overview.svg)

## Strengths

The strength of this project is not “more layers of framing”, but a cleaner end-to-end search MCP experience:

- more auditable search results:
  - `web_search` returns the answer
  - `get_sources` returns sources, trace, and evidence bindings
- stronger aggregation:
  - combines Grok, Tavily, and Firecrawl
  - extra sources are enabled by default to reduce single-answer blind spots
- better handling of complex questions:
  - the six `plan_*` tools support structured decomposition
- more robust retrieval:
  - `web_fetch` uses Tavily Extract first and falls back to Firecrawl Scrape
- easier client adoption:
  - validated with `Codex`, `Claude`, and `Gemini`

## Local Validation

These results come from the local acceptance run on **April 9, 2026**.

### Automated Results

| Item | Result |
|------|--------|
| Regression tests | `22 passed` |
| Real-environment checks | completed |
| Multi-client validation | `Codex / Claude / Gemini` |
| Gating failures | `0` |

### Reproduce

```bash
uv run --with pytest --with pytest-asyncio pytest -q
```

## Installation

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- at least one OpenAI-compatible Grok endpoint
- optional: Tavily / Firecrawl

### Install From Your Fork

If you keep the `grok-with-tavily` branch in your own fork:

```bash
uvx --from git+https://github.com/<yourname>/GrokSearch@grok-with-tavily grok-search
```

For long-term usage, installing as a tool is more stable:

```bash
uv tool install --from git+https://github.com/<yourname>/GrokSearch@grok-with-tavily grok-search
```

### Local Editable Install

```bash
git clone https://github.com/<yourname>/GrokSearch.git
cd GrokSearch
git checkout grok-with-tavily
uv tool install -e .
```

### Important Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROK_API_URL` | Yes | - | OpenAI-compatible Grok endpoint |
| `GROK_API_KEY` | Yes | - | Grok API key |
| `GROK_MODEL` | No | `grok-4.20-beta` | default model |
| `TAVILY_API_URL` | No | `https://api.tavily.com` | Tavily endpoint |
| `TAVILY_API_KEY` | No | - | Tavily key |
| `TAVILY_ENABLED` | No | `true` | enable Tavily |
| `FIRECRAWL_API_URL` | No | `https://api.firecrawl.dev/v2` | Firecrawl endpoint |
| `FIRECRAWL_API_KEY` | No | - | Firecrawl key |
| `GROK_DEBUG` | No | `false` | debug mode |
| `GROK_RETRY_MAX_ATTEMPTS` | No | `3` | max retry attempts |
| `GROK_RETRY_MULTIPLIER` | No | `1` | retry multiplier |
| `GROK_RETRY_MAX_WAIT` | No | `10` | max retry wait seconds |

## Client Setup

### Claude Code

```bash
claude mcp add-json grok-search --scope user '{
  "type": "stdio",
  "command": "uvx",
  "args": [
    "--from",
    "git+https://github.com/<yourname>/GrokSearch@grok-with-tavily",
    "grok-search"
  ],
  "env": {
    "GROK_API_URL": "https://your-grok-endpoint/v1",
    "GROK_API_KEY": "your-grok-key",
    "TAVILY_API_URL": "https://api.tavily.com",
    "TAVILY_API_KEY": "your-tavily-key",
    "TAVILY_ENABLED": "true",
    "FIRECRAWL_API_URL": "https://api.firecrawl.dev/v2",
    "FIRECRAWL_API_KEY": "your-firecrawl-key"
  }
}'
```

### Codex

Example `~/.codex/config.toml`:

```toml
[mcp_servers.grok-search]
type = "stdio"
command = "/home/yourname/.local/bin/grok-search"

[mcp_servers.grok-search.env]
GROK_API_URL = "https://your-grok-endpoint/v1"
GROK_API_KEY = "your-grok-key"
TAVILY_API_URL = "https://api.tavily.com"
TAVILY_API_KEY = "your-tavily-key"
TAVILY_ENABLED = "true"
FIRECRAWL_API_URL = "https://api.firecrawl.dev/v2"
FIRECRAWL_API_KEY = "your-firecrawl-key"
```

### Gemini CLI

Example `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "grok-search": {
      "command": "/home/yourname/.local/bin/grok-search",
      "args": [],
      "env": {
        "GROK_API_URL": "https://your-grok-endpoint/v1",
        "GROK_API_KEY": "your-grok-key",
        "TAVILY_API_URL": "https://api.tavily.com",
        "TAVILY_API_KEY": "your-tavily-key",
        "TAVILY_ENABLED": "true",
        "FIRECRAWL_API_URL": "https://api.firecrawl.dev/v2",
        "FIRECRAWL_API_KEY": "your-firecrawl-key"
      }
    }
  }
}
```

## Tools

This repository currently exposes **13 MCP tools**.

### Search and Retrieval

- `web_search`
- `get_sources`
- `web_fetch`
- `web_map`

### Diagnostics and Control

- `get_config_info`
- `switch_model`
- `toggle_builtin_tools`

### Search Planning

- `plan_intent`
- `plan_complexity`
- `plan_sub_query`
- `plan_search_term`
- `plan_tool_mapping`
- `plan_execution`

## Search Design Principles

This version follows a few explicit rules:

1. **Answer first, then auditability**
   - `web_search` returns the answer
   - `get_sources` returns sources, trace, and evidence bindings

2. **Aggregation by default, not unlimited expansion**
   - default `extra_sources=20`
   - broader questions may trigger follow-up expansion
   - simple questions do not fan out indefinitely

3. **Prefer multi-source support for factual claims**
   - Grok provides the synthesized answer
   - Tavily / Firecrawl add structured supporting sources
   - `evidence_bindings` tries to align claims to more specific sources

4. **Fetching must have a fallback path**
   - Tavily Extract falls back to Firecrawl Scrape

5. **Complex research should be planned**
   - the six `plan_*` tools exist for that purpose

## Development

### Install Dependencies

```bash
uv sync
```

### Run Tests

```bash
uv run --with pytest --with pytest-asyncio pytest -q
```

### Key Files

| Path | Purpose |
|------|---------|
| `src/grok_search/server.py` | MCP entrypoints and main flow |
| `src/grok_search/providers/grok.py` | Grok provider |
| `src/grok_search/sources.py` | source cache and source parsing |
| `tests/test_regressions.py` | regression tests |
| `docs/improvement-workflow.md` | improvement and acceptance workflow |

## FAQ

### Why is `extra_sources=20` the default?

Because this version aims to be an aggregation-oriented search MCP rather than a thin wrapper around a single Grok answer.

### Can I move this to a new machine directly?

Yes. The most reliable path is to push this repo to your own Git repository and install from that fork:

```bash
uv tool install --from git+https://github.com/<yourname>/GrokSearch@grok-with-tavily grok-search
```

### Is this a general agent framework?

No. It is a search-focused MCP toolkit for search, aggregation, fetching, and search planning.
