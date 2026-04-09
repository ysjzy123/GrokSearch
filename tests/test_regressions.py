import json
import inspect
import sys
import asyncio
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grok_search import server
from grok_search.providers.base import BaseSearchProvider
from grok_search.providers.grok import GrokSearchProvider
from grok_search.sources import split_answer_and_sources


@pytest.fixture(autouse=True)
def reset_global_state(monkeypatch, tmp_path):
    monkeypatch.setenv("GROK_API_URL", "https://api.example.test/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_ENABLED", "true")
    monkeypatch.setenv("GROK_SOURCES_CACHE_DIR", str(tmp_path / "sources-cache"))
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.delenv("GROK_MODEL", raising=False)

    server._AVAILABLE_MODELS_CACHE.clear()
    server._SOURCES_CACHE = server.SourcesCache(max_size=256, persist_dir=tmp_path / "sources-cache")
    server.planning_engine._sessions.clear()
    server.config._cached_model = None
    server.config._config_file = tmp_path / "config.json"

    yield

    server._AVAILABLE_MODELS_CACHE.clear()
    server._SOURCES_CACHE = server.SourcesCache(max_size=256, persist_dir=tmp_path / "sources-cache")
    server.planning_engine._sessions.clear()
    server.config._cached_model = None
    server.config._config_file = None


def write_model_config(model: str) -> None:
    server.config.config_file.write_text(
        json.dumps({"model": model}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    server.config._cached_model = None


def test_split_answer_and_sources_strips_think_blocks():
    answer, sources = split_answer_and_sources("<think>trace</think>\nFinal answer")
    assert answer == "Final answer"
    assert sources == []


@pytest.mark.asyncio
async def test_sources_cache_persists_between_instances(tmp_path):
    persist_dir = tmp_path / "persisted-sources"
    first = server.SourcesCache(max_size=2, persist_dir=persist_dir)
    session_id = "session-one"
    payload = [{"url": "https://example.com", "title": "Example"}]

    await first.set(session_id, payload)

    second = server.SourcesCache(max_size=2, persist_dir=persist_dir)
    restored = await second.get(session_id)

    assert restored == payload


@pytest.mark.asyncio
async def test_sources_cache_prunes_old_persisted_entries(tmp_path):
    persist_dir = tmp_path / "persisted-sources"
    cache = server.SourcesCache(max_size=2, persist_dir=persist_dir)

    await cache.set("first", [{"url": "https://example.com/1"}])
    await cache.set("second", [{"url": "https://example.com/2"}])
    await cache.set("third", [{"url": "https://example.com/3"}])

    reloaded = server.SourcesCache(max_size=2, persist_dir=persist_dir)

    assert await reloaded.get("first") is None
    assert await reloaded.get("second") == [{"url": "https://example.com/2"}]
    assert await reloaded.get("third") == [{"url": "https://example.com/3"}]


def test_provider_search_contract_returns_string():
    assert inspect.signature(BaseSearchProvider.search).return_annotation is str
    assert inspect.signature(GrokSearchProvider.search).return_annotation is str


@pytest.mark.asyncio
async def test_provider_injects_time_context_only_for_time_sensitive_queries(monkeypatch):
    captured = {}

    async def fake_exec(self, headers, payload, ctx=None):
        captured["payload"] = payload
        return "ok"

    monkeypatch.setattr(GrokSearchProvider, "_execute_stream_with_retry", fake_exec)
    monkeypatch.setattr(
        "grok_search.providers.grok.get_local_time_info",
        lambda: "[Current Time Context]\n- Date: 2026-04-09",
    )

    provider = GrokSearchProvider("https://api.example.test/v1", "test-key")

    await provider.search("capital of france")
    assert "[Current Time Context]" not in captured["payload"]["messages"][1]["content"]

    await provider.search("latest python release")
    assert "[Current Time Context]" in captured["payload"]["messages"][1]["content"]


@pytest.mark.asyncio
async def test_provider_search_falls_back_to_non_stream_on_connection_failure(monkeypatch):
    captured = {}

    async def fake_stream(self, headers, payload, ctx=None):
        raise __import__("httpx").ConnectError("connect failed")

    async def fake_json(self, headers, payload, ctx=None):
        captured["payload"] = payload
        return "fallback ok"

    monkeypatch.setattr(GrokSearchProvider, "_execute_stream_with_retry", fake_stream)
    monkeypatch.setattr(GrokSearchProvider, "_execute_json_with_retry", fake_json)

    provider = GrokSearchProvider("https://api.example.test/v1", "test-key")
    result = await provider.search("capital of france")

    assert result == "fallback ok"
    assert captured["payload"]["stream"] is False


@pytest.mark.asyncio
async def test_switch_model_rejects_invalid_model_without_writing_config(monkeypatch):
    write_model_config("grok-4.1-fast")

    async def fake_models(*_args):
        return ["grok-4.1-fast", "grok-4.1-mini"]

    monkeypatch.setattr(server, "_get_available_models_cached", fake_models)

    result = json.loads(await server.switch_model("definitely-not-a-real-model"))

    assert result["status"] == "❌ 失败"
    assert result["message"] == "无效模型: definitely-not-a-real-model"
    saved = json.loads(server.config.config_file.read_text(encoding="utf-8"))
    assert saved["model"] == "grok-4.1-fast"


@pytest.mark.asyncio
async def test_toggle_builtin_tools_uses_local_directory_when_no_repo(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    result = json.loads(await server.toggle_builtin_tools("on"))
    settings_path = tmp_path / ".claude" / "settings.json"

    assert result["file"] == str(settings_path)
    assert result["blocked"] is True
    saved = json.loads(settings_path.read_text(encoding="utf-8"))
    assert sorted(saved["permissions"]["deny"]) == ["WebFetch", "WebSearch"]


@pytest.mark.asyncio
async def test_plan_complexity_rejects_invalid_level():
    intent = json.loads(
        await server.plan_intent(
            thought="test",
            core_question="test planning",
            query_type="analytical",
            time_sensitivity="irrelevant",
        )
    )

    result = json.loads(
        await server.plan_complexity(
            session_id=intent["session_id"],
            thought="invalid level",
            level=99,
            estimated_sub_queries=1,
            estimated_tool_calls=1,
            justification="test",
        )
    )

    assert result["error"] == "validation_error"


@pytest.mark.asyncio
async def test_plan_search_term_requires_initial_approach():
    intent = json.loads(
        await server.plan_intent(
            thought="test",
            core_question="test planning",
            query_type="analytical",
            time_sensitivity="irrelevant",
        )
    )

    result = json.loads(
        await server.plan_search_term(
            session_id=intent["session_id"],
            thought="missing approach",
            term="valid short term",
            purpose="sq1",
            round=1,
        )
    )

    assert result["error"] == "validation_error"
    assert result["message"] == "First search term must define approach"


@pytest.mark.asyncio
async def test_plan_search_term_rejects_terms_longer_than_eight_words():
    intent = json.loads(
        await server.plan_intent(
            thought="test",
            core_question="test planning",
            query_type="analytical",
            time_sensitivity="irrelevant",
        )
    )

    result = json.loads(
        await server.plan_search_term(
            session_id=intent["session_id"],
            thought="too long",
            term="this search term is intentionally far longer than eight words",
            purpose="sq1",
            round=1,
            approach="broad_first",
        )
    )

    assert result["error"] == "validation_error"
    assert result["message"] == "Search query must be 8 words or fewer"


@pytest.mark.asyncio
async def test_web_fetch_rejects_invalid_url():
    result = await server.web_fetch("notaurl")
    assert result == "无效URL: notaurl"


@pytest.mark.asyncio
async def test_web_search_uses_default_extra_sources_and_strips_think(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "t-key")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "f-key")

    calls = {}

    class FakeProvider:
        def __init__(self, api_url, api_key, model):
            calls["provider"] = (api_url, api_key, model)

        async def search(self, query, platform):
            calls["query"] = (query, platform)
            return "<think>reasoning</think>\nAnswer body"

        async def describe_url(self, url):
            calls.setdefault("describe", []).append(url)
            return {"title": "Enriched Tavily", "extracts": "Enriched summary", "url": url}

        async def rank_sources(self, query, sources_text, total):
            calls["rank"] = (query, sources_text, total)
            return [2, 1]

    async def fake_tavily(query, max_results=6):
        calls["tavily"] = (query, max_results)
        return [{"url": "https://t.example/item"}]

    async def fake_firecrawl(query, limit=14):
        calls["firecrawl"] = (query, limit)
        return [{"title": "Firecrawl", "url": "https://f.example/item", "description": "summary"}]

    monkeypatch.setattr(server, "GrokSearchProvider", FakeProvider)
    monkeypatch.setattr(server, "_call_tavily_search", fake_tavily)
    monkeypatch.setattr(server, "_call_firecrawl_search", fake_firecrawl)

    result = await server.web_search("capital of france")
    sources = await server.get_sources(result["session_id"])

    assert result["content"] == "Answer body"
    assert result["sources_count"] == 2
    assert calls["tavily"] == ("capital of france", 10)
    assert calls["firecrawl"] == ("capital of france", 10)
    assert calls["describe"] == ["https://t.example/item"]
    assert calls["rank"][2] == 2
    assert sources["sources"][0]["provider"] == "tavily"
    assert sources["sources"][0]["title"] == "Enriched Tavily"
    assert sources["sources"][0]["description"] == "Enriched summary"
    assert sources["sources"][1]["provider"] == "firecrawl"


@pytest.mark.asyncio
async def test_web_search_returns_error_message_when_provider_fails(monkeypatch):
    class FailingProvider:
        def __init__(self, api_url, api_key, model):
            pass

        async def search(self, query, platform):
            raise RuntimeError("boom")

    monkeypatch.setattr(server, "GrokSearchProvider", FailingProvider)

    result = await server.web_search("capital of france", extra_sources=0)

    assert result["content"] == "搜索失败: boom"
    assert result["sources_count"] == 0


@pytest.mark.asyncio
async def test_web_search_expands_broad_queries_and_records_search_trace(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "t-key")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "f-key")

    calls = []

    class FakeProvider:
        def __init__(self, api_url, api_key, model):
            pass

        async def search(self, query, platform):
            return "Rust ownership improves memory safety.\nTokio powers async network services."

        async def describe_url(self, url):
            return {"title": url.rsplit("/", 1)[-1], "extracts": "", "url": url}

        async def rank_sources(self, query, sources_text, total):
            return list(range(1, total + 1))

    async def fake_tavily(query, max_results=6):
        calls.append(("tavily", query, max_results))
        if "roadmap" in query.lower() or "学习路线" in query:
            return [{"url": "https://t.example/roadmap", "title": "Rust Roadmap", "content": "learning roadmap", "facet": "roadmap"}]
        if "best practices" in query.lower() or "最佳实践" in query:
            return [{"url": "https://t.example/best-practices", "title": "Rust Best Practices", "content": "ownership practices", "facet": "best-practices"}]
        return [{"url": "https://t.example/overview", "title": "Rust Overview", "content": "systems programming", "facet": "overview"}]

    async def fake_firecrawl(query, limit=14):
        calls.append(("firecrawl", query, limit))
        if "roadmap" in query.lower() or "学习路线" in query:
            return [{"title": "Rust Project Roadmap", "url": "https://f.example/roadmap", "description": "learning path", "facet": "roadmap"}]
        if "best practices" in query.lower() or "最佳实践" in query:
            return [{"title": "Tokio Runtime Guide", "url": "https://f.example/tokio", "description": "async runtime", "facet": "runtime"}]
        return [{"title": "Rust Ecosystem Overview", "url": "https://f.example/overview", "description": "tooling", "facet": "overview"}]

    monkeypatch.setattr(server, "GrokSearchProvider", FakeProvider)
    monkeypatch.setattr(server, "_call_tavily_search", fake_tavily)
    monkeypatch.setattr(server, "_call_firecrawl_search", fake_firecrawl)

    result = await server.web_search("Rust systems programming learning guide", extra_sources=12)
    cached = await server.get_sources(result["session_id"])

    planned_queries = cached["search_trace"]["planned_queries"]
    provider_counts = cached["search_trace"]["summary"]["provider_counts"]
    facets = {source.get("facet") for source in cached["sources"]}

    assert len(planned_queries) >= 3
    assert cached["search_trace"]["summary"]["expanded"] is True
    assert cached["search_trace"]["summary"]["executed_query_count"] >= 3
    assert provider_counts["tavily"] >= 1
    assert provider_counts["firecrawl"] >= 1
    assert "roadmap" in facets
    assert "best-practices" in facets or "runtime" in facets
    assert any(source.get("query_used") != "Rust systems programming learning guide" for source in cached["sources"])
    assert len({(provider, query) for provider, query, _count in calls}) >= 4
    assert cached["search_trace"]["summary"]["followup_executed"] is True


@pytest.mark.asyncio
async def test_web_search_early_stops_when_initial_support_is_strong(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "t-key")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "f-key")

    calls = []

    class FakeProvider:
        def __init__(self, api_url, api_key, model):
            pass

        async def search(self, query, platform):
            return "Rust ownership improves memory safety.\nTokio powers async network services."

        async def describe_url(self, url):
            return {"title": url.rsplit("/", 1)[-1], "extracts": "", "url": url}

        async def rank_sources(self, query, sources_text, total):
            return list(range(1, total + 1))

    async def fake_tavily(query, max_results=6):
        calls.append(("tavily", query, max_results))
        return [
            {"url": "https://t.example/ownership", "title": "Rust Ownership", "content": "ownership memory safety", "facet": "ownership"},
            {"url": "https://t.example/roadmap", "title": "Rust Roadmap", "content": "learning roadmap", "facet": "roadmap"},
        ]

    async def fake_firecrawl(query, limit=14):
        calls.append(("firecrawl", query, limit))
        return [
            {"title": "Tokio Runtime Guide", "url": "https://f.example/tokio", "description": "tokio async runtime network services", "facet": "runtime"},
            {"title": "Rust Book Summary", "url": "https://f.example/book", "description": "official guide and examples", "facet": "overview"},
        ]

    monkeypatch.setattr(server, "GrokSearchProvider", FakeProvider)
    monkeypatch.setattr(server, "_call_tavily_search", fake_tavily)
    monkeypatch.setattr(server, "_call_firecrawl_search", fake_firecrawl)

    result = await server.web_search("Rust systems programming learning guide", extra_sources=12)
    cached = await server.get_sources(result["session_id"])

    summary = cached["search_trace"]["summary"]
    phases = {phase["name"]: phase for phase in cached["search_trace"]["phases"]}

    assert summary["early_stopped"] is True
    assert summary["followup_executed"] is False
    assert summary["budget_used"] < summary["budget_requested"]
    assert phases["expansion"]["skipped"] is True
    assert phases["expansion"]["reason"] == "initial_support_sufficient"
    assert len(calls) == 2
    assert all(query == "Rust systems programming learning guide" for _provider, query, _count in calls)


@pytest.mark.asyncio
async def test_get_sources_returns_evidence_bindings(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "t-key")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "f-key")

    class FakeProvider:
        def __init__(self, api_url, api_key, model):
            pass

        async def search(self, query, platform):
            return "Rust ownership improves memory safety.\nTokio powers async network services."

        async def describe_url(self, url):
            return {"title": url.rsplit("/", 1)[-1], "extracts": "", "url": url}

        async def rank_sources(self, query, sources_text, total):
            return list(range(1, total + 1))

    async def fake_tavily(query, max_results=6):
        return [{"url": "https://t.example/ownership", "title": "Rust ownership guide", "content": "ownership memory safety"}]

    async def fake_firecrawl(query, limit=14):
        return [{"title": "Tokio async runtime", "url": "https://f.example/tokio", "description": "tokio async runtime network services"}]

    monkeypatch.setattr(server, "GrokSearchProvider", FakeProvider)
    monkeypatch.setattr(server, "_call_tavily_search", fake_tavily)
    monkeypatch.setattr(server, "_call_firecrawl_search", fake_firecrawl)

    result = await server.web_search("Rust async services guide", extra_sources=4)
    cached = await server.get_sources(result["session_id"])

    bindings = cached["evidence_bindings"]
    by_claim = {item["claim"]: item["sources"] for item in bindings}

    assert len(bindings) >= 2
    assert any(source["url"] == "https://t.example/ownership" for source in by_claim["Rust ownership improves memory safety."])
    assert any(source["url"] == "https://f.example/tokio" for source in by_claim["Tokio powers async network services."])


@pytest.mark.asyncio
async def test_evidence_bindings_filter_generic_decoys(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "t-key")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "f-key")

    class FakeProvider:
        def __init__(self, api_url, api_key, model):
            pass

        async def search(self, query, platform):
            return "Rust ownership improves memory safety.\nTokio powers async network services."

        async def describe_url(self, url):
            return {"title": url.rsplit("/", 1)[-1], "extracts": "", "url": url}

        async def rank_sources(self, query, sources_text, total):
            return list(range(1, total + 1))

    async def fake_tavily(query, max_results=6):
        return [
            {"url": "https://t.example/ownership", "title": "Rust ownership guide", "content": "ownership memory safety"},
            {"url": "https://t.example/generic-memory", "title": "Memory safety overview", "content": "memory safety techniques in systems languages"},
        ]

    async def fake_firecrawl(query, limit=14):
        return [
            {"title": "Tokio async runtime", "url": "https://f.example/tokio", "description": "tokio async runtime network services"},
            {"title": "Async network services overview", "url": "https://f.example/generic-async", "description": "patterns for async network services"},
        ]

    monkeypatch.setattr(server, "GrokSearchProvider", FakeProvider)
    monkeypatch.setattr(server, "_call_tavily_search", fake_tavily)
    monkeypatch.setattr(server, "_call_firecrawl_search", fake_firecrawl)

    result = await server.web_search("Rust async services guide", extra_sources=4)
    cached = await server.get_sources(result["session_id"])

    bindings = cached["evidence_bindings"]
    by_claim = {item["claim"]: item["sources"] for item in bindings}

    assert by_claim["Rust ownership improves memory safety."][0]["url"] == "https://t.example/ownership"
    assert by_claim["Tokio powers async network services."][0]["url"] == "https://f.example/tokio"
    assert all(source["url"] != "https://t.example/generic-memory" for source in by_claim["Rust ownership improves memory safety."])
    assert all(source["url"] != "https://f.example/generic-async" for source in by_claim["Tokio powers async network services."])


@pytest.mark.asyncio
async def test_enrich_and_rank_sources_times_out_gracefully(monkeypatch):
    class SlowProvider:
        async def describe_url(self, url):
            await asyncio.sleep(0.05)
            return {"title": "slow", "extracts": "slow", "url": url}

        async def rank_sources(self, query, sources_text, total):
            await asyncio.sleep(0.05)
            return [2, 1]

    monkeypatch.setattr(server, "_SOURCE_ENRICH_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(server, "_SOURCE_RANK_TIMEOUT_SECONDS", 0.01)

    sources = [
        {"url": "https://example.com/a", "provider": "tavily"},
        {"url": "https://example.com/b", "provider": "firecrawl", "title": "B", "description": "Desc B"},
    ]

    result = await server._enrich_and_rank_sources("example query", SlowProvider(), sources)

    assert result == sources
