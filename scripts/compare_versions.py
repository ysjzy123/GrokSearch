#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import io
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_REF = "origin/grok-with-tavily"
ENV_KEYS = [
    "GROK_API_URL",
    "GROK_API_KEY",
    "TAVILY_API_URL",
    "TAVILY_API_KEY",
    "TAVILY_ENABLED",
    "FIRECRAWL_API_URL",
    "FIRECRAWL_API_KEY",
]
LIVE_ENV_KEYS = ENV_KEYS + [
    "GUDA_API_KEY",
    "GUDA_BASE_URL",
    "GROK_MODEL",
]


@dataclass
class Probe:
    category: str
    name: str
    passed: bool
    points: int
    details: str
    gating: bool = False


def _purge_grok_modules() -> None:
    for name in list(sys.modules):
        if name == "grok_search" or name.startswith("grok_search."):
            del sys.modules[name]


def _load_target(repo_path: Path):
    _purge_grok_modules()
    src = str(repo_path / "src")
    if src in sys.path:
        sys.path.remove(src)
    sys.path.insert(0, src)

    server = importlib.import_module("grok_search.server")
    providers_base = importlib.import_module("grok_search.providers.base")
    providers_grok = importlib.import_module("grok_search.providers.grok")
    sources = importlib.import_module("grok_search.sources")
    return server, providers_base, providers_grok, sources


def _capture_env(keys: list[str]) -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in keys}


def _restore_env(snapshot: dict[str, str | None]) -> None:
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _seed_deterministic_env() -> None:
    os.environ["GROK_API_URL"] = "https://api.example.test/v1"
    os.environ["GROK_API_KEY"] = "test-key"
    os.environ["TAVILY_ENABLED"] = "true"
    os.environ.pop("TAVILY_API_KEY", None)
    os.environ.pop("FIRECRAWL_API_KEY", None)
    os.environ.pop("GROK_MODEL", None)


def _reset_target_state(server_module, tmp_root: Path) -> None:
    server_module._AVAILABLE_MODELS_CACHE.clear()
    server_module._SOURCES_CACHE = server_module.SourcesCache(max_size=256)
    server_module.planning_engine._sessions.clear()
    server_module.config._cached_model = None
    server_module.config._config_file = tmp_root / "config.json"


async def _probe_validation(server_module, tmp_root: Path) -> list[Probe]:
    probes: list[Probe] = []

    intent = json.loads(
        await server_module.plan_intent(
            thought="test",
            core_question="test planning",
            query_type="analytical",
            time_sensitivity="irrelevant",
        )
    )
    session_id = intent["session_id"]

    invalid_level = json.loads(
        await server_module.plan_complexity(
            session_id=session_id,
            thought="invalid",
            level=99,
            estimated_sub_queries=1,
            estimated_tool_calls=1,
            justification="test",
        )
    )
    probes.append(
        Probe(
            "validation",
            "invalid_complexity_rejected",
            invalid_level.get("error") == "validation_error",
            4,
            invalid_level.get("message") or invalid_level.get("error", ""),
            gating=True,
        )
    )

    missing_approach = json.loads(
        await server_module.plan_search_term(
            session_id=session_id,
            thought="missing approach",
            term="valid short term",
            purpose="sq1",
            round=1,
        )
    )
    probes.append(
        Probe(
            "validation",
            "initial_search_term_requires_approach",
            missing_approach.get("message") == "First search term must define approach",
            4,
            missing_approach.get("message") or missing_approach.get("error", ""),
            gating=True,
        )
    )

    invalid_fetch = await server_module.web_fetch("notaurl")
    probes.append(
        Probe(
            "validation",
            "invalid_fetch_url_rejected",
            invalid_fetch == "无效URL: notaurl",
            4,
            invalid_fetch,
            gating=True,
        )
    )

    async def fake_models(*_args):
        return ["grok-4.1-fast", "grok-4.1-mini"]

    original = server_module._get_available_models_cached
    server_module._get_available_models_cached = fake_models
    try:
        server_module.config.config_file.write_text(
            json.dumps({"model": "grok-4.1-fast"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        server_module.config._cached_model = None
        invalid_model = json.loads(await server_module.switch_model("definitely-not-a-real-model"))
        saved = json.loads(server_module.config.config_file.read_text(encoding="utf-8"))
        passed = (
            invalid_model.get("status") == "❌ 失败"
            and saved.get("model") == "grok-4.1-fast"
        )
        probes.append(
            Probe(
                "validation",
                "invalid_model_not_persisted",
                passed,
                4,
                invalid_model.get("message", ""),
                gating=True,
            )
        )
    finally:
        server_module._get_available_models_cached = original

    expected_settings = tmp_root / "toggle-probe" / ".claude" / "settings.json"
    expected_settings.parent.parent.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    os.chdir(expected_settings.parent.parent)
    try:
        toggle_status = json.loads(await server_module.toggle_builtin_tools("status"))
        try:
            toggle_on = json.loads(await server_module.toggle_builtin_tools("on"))
        except Exception as exc:
            toggle_on = {"error": str(exc)}
    finally:
        os.chdir(previous_cwd)

    probes.append(
        Probe(
            "validation",
            "toggle_builtin_tools_uses_local_workspace",
            toggle_status.get("file") == str(expected_settings)
            and toggle_on.get("file") == str(expected_settings)
            and toggle_on.get("blocked") is True
            and expected_settings.exists(),
            4,
            json.dumps(
                {"status": toggle_status, "on": toggle_on},
                ensure_ascii=False,
            ),
            gating=True,
        )
    )

    return probes


async def _probe_source_quality(server_module, sources_module) -> list[Probe]:
    probes: list[Probe] = []

    answer, extracted = sources_module.split_answer_and_sources("<think>trace</think>\nAnswer body")
    probes.append(
        Probe(
            "source_quality",
            "think_blocks_stripped",
            answer == "Answer body" and extracted == [],
            5,
            answer,
            gating=True,
        )
    )

    calls: dict[str, Any] = {}
    os.environ["TAVILY_API_KEY"] = "t-key"
    os.environ["FIRECRAWL_API_KEY"] = "f-key"

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

    original_provider = server_module.GrokSearchProvider
    original_tavily = server_module._call_tavily_search
    original_firecrawl = server_module._call_firecrawl_search
    server_module.GrokSearchProvider = FakeProvider
    server_module._call_tavily_search = fake_tavily
    server_module._call_firecrawl_search = fake_firecrawl

    try:
        result = await server_module.web_search("capital of france")
        cached = await server_module.get_sources(result["session_id"])
    finally:
        server_module.GrokSearchProvider = original_provider
        server_module._call_tavily_search = original_tavily
        server_module._call_firecrawl_search = original_firecrawl
        os.environ.pop("TAVILY_API_KEY", None)
        os.environ.pop("FIRECRAWL_API_KEY", None)

    probes.append(
        Probe(
            "source_quality",
            "default_extra_sources_enabled",
            result["sources_count"] >= 2,
            5,
            f"sources_count={result['sources_count']}",
            gating=True,
        )
    )
    probes.append(
        Probe(
            "source_quality",
            "balanced_extra_source_budget",
            calls.get("tavily") == ("capital of france", 10)
            and calls.get("firecrawl") == ("capital of france", 10),
            5,
            f"tavily={calls.get('tavily')} firecrawl={calls.get('firecrawl')}",
            gating=True,
        )
    )
    probes.append(
        Probe(
            "source_quality",
            "missing_metadata_enriched",
            calls.get("describe") == ["https://t.example/item"]
            and cached["sources"][0].get("title") == "Enriched Tavily",
            5,
            json.dumps(cached["sources"], ensure_ascii=False),
        )
    )
    probes.append(
        Probe(
            "source_quality",
            "sources_ranked",
            bool(calls.get("rank")) and cached["sources"][0].get("provider") == "tavily",
            5,
            json.dumps(cached["sources"], ensure_ascii=False),
        )
    )

    return probes


async def _probe_query_quality(providers_grok) -> list[Probe]:
    probes: list[Probe] = []
    captured: dict[str, Any] = {}

    async def fake_exec(self, headers, payload, ctx=None):
        captured["payload"] = payload
        return "ok"

    original_exec = providers_grok.GrokSearchProvider._execute_stream_with_retry
    original_time = providers_grok.get_local_time_info
    providers_grok.GrokSearchProvider._execute_stream_with_retry = fake_exec
    providers_grok.get_local_time_info = lambda: "[Current Time Context]\n- Date: 2026-04-09"

    try:
        provider = providers_grok.GrokSearchProvider("https://api.example.test/v1", "test-key")
        await provider.search("capital of france")
        no_time = captured["payload"]["messages"][1]["content"]
        await provider.search("latest python release")
        with_time = captured["payload"]["messages"][1]["content"]
    finally:
        providers_grok.GrokSearchProvider._execute_stream_with_retry = original_exec
        providers_grok.get_local_time_info = original_time

    probes.append(
        Probe(
            "query_quality",
            "timeless_queries_skip_time_context",
            "[Current Time Context]" not in no_time,
            8,
            no_time,
            gating=True,
        )
    )
    probes.append(
        Probe(
            "query_quality",
            "time_sensitive_queries_include_time_context",
            "[Current Time Context]" in with_time,
            7,
            with_time,
            gating=True,
        )
    )
    return probes


async def _probe_resilience(server_module) -> list[Probe]:
    probes: list[Probe] = []

    class FailingProvider:
        def __init__(self, api_url, api_key, model):
            pass

        async def search(self, query, platform):
            raise RuntimeError("boom")

    original_provider = server_module.GrokSearchProvider
    server_module.GrokSearchProvider = FailingProvider
    try:
        result = await server_module.web_search("capital of france", extra_sources=0)
    finally:
        server_module.GrokSearchProvider = original_provider

    probes.append(
        Probe(
            "resilience",
            "provider_error_surfaces_to_user",
            result["content"] == "搜索失败: boom" and result["sources_count"] == 0,
            8,
            json.dumps(result, ensure_ascii=False),
            gating=True,
        )
    )

    if hasattr(server_module, "_split_extra_sources_budget"):
        budget = server_module._split_extra_sources_budget(20, True, True)
        budget_passed = budget[0] > 0 and budget[1] > 0
        budget_details = f"budget={budget}"
    else:
        budget = None
        budget_passed = False
        budget_details = "missing _split_extra_sources_budget helper"

    probes.append(
        Probe(
            "resilience",
            "source_budget_uses_both_backends",
            budget_passed,
            7,
            budget_details,
            gating=True,
        )
    )

    return probes


async def _probe_cache_quality(sources_module, tmp_root: Path) -> list[Probe]:
    probes: list[Probe] = []

    cache_dir = tmp_root / "persisted-sources"
    cache_cls = sources_module.SourcesCache
    session_id = "score-session"
    payload = [{"url": "https://example.com/cache", "title": "Cache Example"}]

    try:
        cache = cache_cls(max_size=2, persist_dir=cache_dir)
    except TypeError as exc:
        probes.append(
            Probe(
                "cache_quality",
                "sources_cache_supports_persistence",
                False,
                5,
                str(exc),
                gating=True,
            )
        )
        probes.append(
            Probe(
                "cache_quality",
                "sources_cache_prunes_persisted_entries",
                False,
                5,
                str(exc),
                gating=True,
            )
        )
        return probes

    await cache.set(session_id, payload)
    reloaded = cache_cls(max_size=2, persist_dir=cache_dir)
    restored = await reloaded.get(session_id)
    probes.append(
        Probe(
            "cache_quality",
            "sources_cache_supports_persistence",
            restored == payload,
            5,
            json.dumps(restored or [], ensure_ascii=False),
            gating=True,
        )
    )

    await reloaded.set("second", [{"url": "https://example.com/2"}])
    await reloaded.set("third", [{"url": "https://example.com/3"}])
    newest = cache_cls(max_size=2, persist_dir=cache_dir)
    oldest = await newest.get(session_id)
    latest = await newest.get("third")
    probes.append(
        Probe(
            "cache_quality",
            "sources_cache_prunes_persisted_entries",
            oldest is None and latest == [{"url": "https://example.com/3"}],
            5,
            json.dumps(
                {"oldest": oldest, "latest": latest},
                ensure_ascii=False,
            ),
            gating=True,
        )
    )

    return probes


async def _run_broad_search_fixture(server_module) -> tuple[dict[str, Any], dict[str, Any], list[tuple[str, str, int]]]:
    calls: list[tuple[str, str, int]] = []
    os.environ["TAVILY_API_KEY"] = "t-key"
    os.environ["FIRECRAWL_API_KEY"] = "f-key"

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

    original_provider = server_module.GrokSearchProvider
    original_tavily = server_module._call_tavily_search
    original_firecrawl = server_module._call_firecrawl_search
    server_module.GrokSearchProvider = FakeProvider
    server_module._call_tavily_search = fake_tavily
    server_module._call_firecrawl_search = fake_firecrawl

    try:
        result = await server_module.web_search("Rust systems programming learning guide", extra_sources=12)
        cached = await server_module.get_sources(result["session_id"])
    finally:
        server_module.GrokSearchProvider = original_provider
        server_module._call_tavily_search = original_tavily
        server_module._call_firecrawl_search = original_firecrawl
        os.environ.pop("TAVILY_API_KEY", None)
        os.environ.pop("FIRECRAWL_API_KEY", None)

    return result, cached, calls


async def _run_early_stop_fixture(server_module) -> tuple[dict[str, Any], dict[str, Any], list[tuple[str, str, int]]]:
    calls: list[tuple[str, str, int]] = []
    os.environ["TAVILY_API_KEY"] = "t-key"
    os.environ["FIRECRAWL_API_KEY"] = "f-key"

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

    original_provider = server_module.GrokSearchProvider
    original_tavily = server_module._call_tavily_search
    original_firecrawl = server_module._call_firecrawl_search
    server_module.GrokSearchProvider = FakeProvider
    server_module._call_tavily_search = fake_tavily
    server_module._call_firecrawl_search = fake_firecrawl

    try:
        result = await server_module.web_search("Rust systems programming learning guide", extra_sources=12)
        cached = await server_module.get_sources(result["session_id"])
    finally:
        server_module.GrokSearchProvider = original_provider
        server_module._call_tavily_search = original_tavily
        server_module._call_firecrawl_search = original_firecrawl
        os.environ.pop("TAVILY_API_KEY", None)
        os.environ.pop("FIRECRAWL_API_KEY", None)

    return result, cached, calls


async def _probe_search_coverage(server_module) -> list[Probe]:
    probes: list[Probe] = []
    _result, cached, _calls = await _run_broad_search_fixture(server_module)

    trace = cached.get("search_trace", {})
    summary = trace.get("summary", {})
    facets = {source.get("facet") for source in cached.get("sources", [])}

    probes.append(
        Probe(
            "search_coverage",
            "broad_queries_expand_into_multiple_searches",
            len(trace.get("planned_queries", [])) >= 3 and summary.get("expanded") is True,
            5,
            json.dumps(trace, ensure_ascii=False),
            gating=True,
        )
    )
    probes.append(
        Probe(
            "search_coverage",
            "expanded_searches_capture_multiple_facets",
            summary.get("executed_query_count", 0) >= 3
            and "roadmap" in facets
            and ("best-practices" in facets or "runtime" in facets),
            5,
            json.dumps(cached.get("sources", []), ensure_ascii=False),
            gating=True,
        )
    )

    return probes


async def _probe_aggregation_breadth(server_module) -> list[Probe]:
    probes: list[Probe] = []
    _result, cached, calls = await _run_broad_search_fixture(server_module)

    trace = cached.get("search_trace", {})
    summary = trace.get("summary", {})
    query_used = {source.get("query_used") for source in cached.get("sources", []) if source.get("query_used")}

    probes.append(
        Probe(
            "aggregation_breadth",
            "multi_engine_aggregation_recorded_in_trace",
            summary.get("provider_counts", {}).get("tavily", 0) >= 1
            and summary.get("provider_counts", {}).get("firecrawl", 0) >= 1,
            5,
            json.dumps(trace, ensure_ascii=False),
            gating=True,
        )
    )
    probes.append(
        Probe(
            "aggregation_breadth",
            "aggregated_sources_preserve_query_metadata",
            len(query_used) >= 2 and len({(provider, query) for provider, query, _count in calls}) >= 4,
            5,
            json.dumps(cached.get("sources", []), ensure_ascii=False),
            gating=True,
        )
    )

    return probes


async def _probe_evidence_binding(server_module) -> list[Probe]:
    probes: list[Probe] = []
    os.environ["TAVILY_API_KEY"] = "t-key"
    os.environ["FIRECRAWL_API_KEY"] = "f-key"

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

    original_provider = server_module.GrokSearchProvider
    original_tavily = server_module._call_tavily_search
    original_firecrawl = server_module._call_firecrawl_search
    server_module.GrokSearchProvider = FakeProvider
    server_module._call_tavily_search = fake_tavily
    server_module._call_firecrawl_search = fake_firecrawl

    try:
        result = await server_module.web_search("Rust async services guide", extra_sources=4)
        cached = await server_module.get_sources(result["session_id"])
    finally:
        server_module.GrokSearchProvider = original_provider
        server_module._call_tavily_search = original_tavily
        server_module._call_firecrawl_search = original_firecrawl
        os.environ.pop("TAVILY_API_KEY", None)
        os.environ.pop("FIRECRAWL_API_KEY", None)

    bindings = cached.get("evidence_bindings", [])
    binding_map = {
        item.get("claim"): [source.get("url") for source in item.get("sources", [])]
        for item in bindings
        if isinstance(item, dict)
    }

    probes.append(
        Probe(
            "evidence_binding",
            "answer_claims_receive_source_bindings",
            len(bindings) >= 2,
            5,
            json.dumps(bindings, ensure_ascii=False),
            gating=True,
        )
    )
    probes.append(
        Probe(
            "evidence_binding",
            "bindings_match_claim_specific_sources",
            "https://t.example/ownership" in binding_map.get("Rust ownership improves memory safety.", [])
            and "https://f.example/tokio" in binding_map.get("Tokio powers async network services.", []),
            5,
            json.dumps(bindings, ensure_ascii=False),
            gating=True,
        )
    )

    return probes


async def _probe_citation_precision(server_module) -> list[Probe]:
    probes: list[Probe] = []
    os.environ["TAVILY_API_KEY"] = "t-key"
    os.environ["FIRECRAWL_API_KEY"] = "f-key"

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

    original_provider = server_module.GrokSearchProvider
    original_tavily = server_module._call_tavily_search
    original_firecrawl = server_module._call_firecrawl_search
    server_module.GrokSearchProvider = FakeProvider
    server_module._call_tavily_search = fake_tavily
    server_module._call_firecrawl_search = fake_firecrawl

    try:
        result = await server_module.web_search("Rust async services guide", extra_sources=4)
        cached = await server_module.get_sources(result["session_id"])
    finally:
        server_module.GrokSearchProvider = original_provider
        server_module._call_tavily_search = original_tavily
        server_module._call_firecrawl_search = original_firecrawl
        os.environ.pop("TAVILY_API_KEY", None)
        os.environ.pop("FIRECRAWL_API_KEY", None)

    bindings = cached.get("evidence_bindings", [])
    top_binding_map = {
        item.get("claim"): (item.get("sources") or [{}])[0].get("url")
        for item in bindings
        if isinstance(item, dict) and item.get("sources")
    }
    all_binding_urls = {
        source.get("url")
        for item in bindings
        if isinstance(item, dict)
        for source in item.get("sources", [])
        if isinstance(source, dict) and source.get("url")
    }

    probes.append(
        Probe(
            "citation_precision",
            "claim_bindings_prioritize_specific_top_source",
            top_binding_map.get("Rust ownership improves memory safety.") == "https://t.example/ownership"
            and top_binding_map.get("Tokio powers async network services.") == "https://f.example/tokio",
            5,
            json.dumps(bindings, ensure_ascii=False),
            gating=True,
        )
    )
    probes.append(
        Probe(
            "citation_precision",
            "generic_decoys_filtered_from_claim_bindings",
            "https://t.example/generic-memory" not in all_binding_urls
            and "https://f.example/generic-async" not in all_binding_urls,
            5,
            json.dumps(bindings, ensure_ascii=False),
            gating=True,
        )
    )

    return probes


async def _probe_efficiency(server_module) -> list[Probe]:
    probes: list[Probe] = []
    _early_result, early_cached, early_calls = await _run_early_stop_fixture(server_module)
    _broad_result, broad_cached, _broad_calls = await _run_broad_search_fixture(server_module)

    early_trace = early_cached.get("search_trace", {})
    early_summary = early_trace.get("summary", {})
    early_phases = {phase.get("name"): phase for phase in early_trace.get("phases", []) if isinstance(phase, dict)}
    broad_trace = broad_cached.get("search_trace", {})
    broad_summary = broad_trace.get("summary", {})
    broad_phases = {phase.get("name"): phase for phase in broad_trace.get("phases", []) if isinstance(phase, dict)}

    probes.append(
        Probe(
            "efficiency",
            "fanout_budget_respected",
            early_summary.get("external_task_count") == 2 and len(early_calls) == 2,
            5,
            json.dumps({"summary": early_summary, "calls": early_calls}, ensure_ascii=False),
            gating=True,
        )
    )
    probes.append(
        Probe(
            "efficiency",
            "early_stop_preserves_unused_budget",
            early_summary.get("early_stopped") is True
            and early_summary.get("budget_unused", 0) > 0
            and early_phases.get("expansion", {}).get("skipped") is True,
            5,
            json.dumps({"summary": early_summary, "phases": early_trace.get("phases", [])}, ensure_ascii=False),
            gating=True,
        )
    )
    probes.append(
        Probe(
            "efficiency",
            "expansion_only_when_initial_coverage_is_weak",
            broad_summary.get("followup_executed") is True
            and broad_summary.get("early_stopped") is False
            and early_summary.get("followup_executed") is False,
            5,
            json.dumps({"broad": broad_summary, "early": early_summary}, ensure_ascii=False),
            gating=True,
        )
    )
    probes.append(
        Probe(
            "efficiency",
            "followup_budget_delivers_new_sources",
            broad_phases.get("expansion", {}).get("budget_used", 0) > 0
            and broad_phases.get("expansion", {}).get("source_count", 0) > 0,
            5,
            json.dumps({"summary": broad_summary, "phases": broad_trace.get("phases", [])}, ensure_ascii=False),
            gating=True,
        )
    )

    return probes


async def _probe_live_retrieval_robustness(server_module) -> list[Probe]:
    try:
        has_fetch_backend = bool(server_module.config.tavily_api_key) or bool(server_module.config.firecrawl_api_key)
    except Exception:
        return []
    if not has_fetch_backend:
        return []

    probes: list[Probe] = []
    github_url = "https://github.com/ckckck/UltimateSearchSkill"
    linux_do_url = "https://linux.do/t/topic/1674101?u=ysjzy"

    github_started = asyncio.get_running_loop().time()
    github_content = await server_module.web_fetch(github_url)
    github_elapsed = asyncio.get_running_loop().time() - github_started

    linux_started = asyncio.get_running_loop().time()
    linux_content = await server_module.web_fetch(linux_do_url)
    linux_elapsed = asyncio.get_running_loop().time() - linux_started

    probes.append(
        Probe(
            "retrieval_robustness",
            "github_repo_fetch_returns_substantial_content",
            isinstance(github_content, str)
            and not github_content.startswith(("配置错误:", "提取失败:", "无效URL:"))
            and len(github_content) >= 500,
            6,
            f"len={len(github_content) if isinstance(github_content, str) else 0} elapsed={github_elapsed:.2f}s",
            gating=False,
        )
    )
    probes.append(
        Probe(
            "retrieval_robustness",
            "linux_do_topic_fetch_returns_substantial_content",
            isinstance(linux_content, str)
            and not linux_content.startswith(("配置错误:", "提取失败:", "无效URL:"))
            and len(linux_content) >= 500,
            8,
            f"len={len(linux_content) if isinstance(linux_content, str) else 0} elapsed={linux_elapsed:.2f}s",
            gating=False,
        )
    )

    if bool(server_module.config.tavily_api_key):
        map_started = asyncio.get_running_loop().time()
        github_map = await server_module.web_map(github_url, limit=10, max_depth=1, max_breadth=8, timeout=60)
        map_elapsed = asyncio.get_running_loop().time() - map_started
        map_len = len(github_map) if isinstance(github_map, str) else 0
        probes.append(
            Probe(
                "retrieval_robustness",
                "github_repo_map_discovers_multiple_links",
                isinstance(github_map, str)
                and not github_map.startswith(("配置错误:", "无效URL:"))
                and map_len >= 200,
                6,
                f"len={map_len} elapsed={map_elapsed:.2f}s",
                gating=False,
            )
        )

    return probes


async def _probe_live_latency_and_cost(server_module) -> list[Probe]:
    try:
        server_module.config._cached_model = None
        _ = server_module.config.grok_api_url
        _ = server_module.config.grok_api_key
    except Exception:
        return []

    probes: list[Probe] = []
    loop = asyncio.get_running_loop()

    easy_started = loop.time()
    easy = await server_module.web_search("What is the capital of France?", extra_sources=12)
    easy_elapsed = loop.time() - easy_started
    easy_cached = await server_module.get_sources(easy["session_id"])
    easy_summary = easy_cached.get("search_trace", {}).get("summary", {})

    hard_started = loop.time()
    hard = await server_module.web_search("Rust systems programming learning guide", extra_sources=12)
    hard_elapsed = loop.time() - hard_started
    hard_cached = await server_module.get_sources(hard["session_id"])
    hard_summary = hard_cached.get("search_trace", {}).get("summary", {})

    probes.append(
        Probe(
            "live_efficiency",
            "easy_query_uses_less_than_max_budget",
            easy_summary.get("budget_used", easy_summary.get("budget_requested", 0)) < easy_summary.get("budget_requested", 0),
            5,
            f"elapsed={easy_elapsed:.2f}s summary={json.dumps(easy_summary, ensure_ascii=False)}",
            gating=False,
        )
    )
    probes.append(
        Probe(
            "live_efficiency",
            "easy_query_latency_remains_reasonable",
            easy_elapsed <= 18.0,
            5,
            f"elapsed={easy_elapsed:.2f}s",
            gating=False,
        )
    )
    probes.append(
        Probe(
            "live_efficiency",
            "hard_query_followup_budget_produces_additional_coverage",
            hard_summary.get("followup_executed") is True and hard.get("sources_count", 0) >= 2,
            5,
            f"elapsed={hard_elapsed:.2f}s summary={json.dumps(hard_summary, ensure_ascii=False)}",
            gating=False,
        )
    )
    probes.append(
        Probe(
            "live_efficiency",
            "hard_query_latency_remains_bounded",
            hard_elapsed <= 30.0,
            5,
            f"elapsed={hard_elapsed:.2f}s",
            gating=False,
        )
    )

    return probes


def _probe_contract(providers_base, providers_grok, repo_path: Path) -> list[Probe]:
    probes: list[Probe] = []

    probes.append(
        Probe(
            "contract",
            "provider_search_returns_string",
            inspect.signature(providers_base.BaseSearchProvider.search).return_annotation is str
            and inspect.signature(providers_grok.GrokSearchProvider.search).return_annotation is str,
            5,
            "search annotations checked",
            gating=True,
        )
    )
    probes.append(
        Probe(
            "contract",
            "provider_fetch_contract_defined",
            hasattr(providers_base.BaseSearchProvider, "fetch"),
            5,
            "fetch annotation exists on base provider",
            gating=True,
        )
    )

    return probes


def _probe_engineering(repo_path: Path) -> list[Probe]:
    probes: list[Probe] = []

    gitignore = repo_path / ".gitignore"
    tests_dir = repo_path / "tests"
    gitignore_text = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""

    probes.append(
        Probe(
            "engineering",
            "regression_tests_present",
            tests_dir.exists(),
            8,
            str(tests_dir),
            gating=True,
        )
    )
    probes.append(
        Probe(
            "engineering",
            "local_artifacts_ignored",
            ".venv/" in gitignore_text and "uv.lock" in gitignore_text and "src/*.egg-info/" in gitignore_text,
            7,
            gitignore_text,
        )
    )

    return probes


async def _probe_live(server_module) -> list[Probe]:
    try:
        server_module.config._cached_model = None
        _ = server_module.config.grok_api_url
        _ = server_module.config.grok_api_key
    except Exception:
        return []

    probes: list[Probe] = []
    non_time = await server_module.web_search("What is the capital of France?")
    recent = await server_module.web_search("When was Python 3.13 released?", extra_sources=0)
    non_time_sources = await server_module.get_sources(non_time["session_id"])

    probes.append(
        Probe(
            "live",
            "non_time_no_think_leak",
            "<think>" not in non_time["content"],
            8,
            non_time["content"][:160],
            gating=True,
        )
    )
    probes.append(
        Probe(
            "live",
            "recent_no_think_leak",
            "<think>" not in recent["content"],
            6,
            recent["content"][:160],
            gating=True,
        )
    )
    probes.append(
        Probe(
            "live",
            "non_time_sources_available",
            non_time["sources_count"] > 0 and non_time_sources["sources_count"] > 0,
            6,
            f"response={non_time['sources_count']} cached={non_time_sources['sources_count']}",
            gating=True,
        )
    )
    return probes


def _failed_gating_probes(categories: dict[str, dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for category_name, bucket in categories.items():
        for probe in bucket.get("probes", []):
            if probe.get("gating") and not probe.get("passed"):
                failures.append(f"{category_name}:{probe['name']}")
    return failures


async def score_repo(repo_path: Path, include_live: bool) -> dict[str, Any]:
    score_tmp = Path(tempfile.mkdtemp(prefix="grok-score-"))
    env_snapshot = _capture_env(LIVE_ENV_KEYS)
    try:
        server_module, providers_base, providers_grok, sources_module = _load_target(repo_path)
        _seed_deterministic_env()
        _reset_target_state(server_module, score_tmp)

        probes: list[Probe] = []
        probes.extend(_probe_contract(providers_base, providers_grok, repo_path))
        probes.extend(await _probe_validation(server_module, score_tmp))
        probes.extend(await _probe_source_quality(server_module, sources_module))
        probes.extend(await _probe_query_quality(providers_grok))
        probes.extend(await _probe_resilience(server_module))
        probes.extend(await _probe_cache_quality(sources_module, score_tmp))
        probes.extend(await _probe_search_coverage(server_module))
        probes.extend(await _probe_aggregation_breadth(server_module))
        probes.extend(await _probe_evidence_binding(server_module))
        probes.extend(await _probe_citation_precision(server_module))
        probes.extend(await _probe_efficiency(server_module))
        probes.extend(_probe_engineering(repo_path))
        if include_live:
            _restore_env(env_snapshot)
            _reset_target_state(server_module, score_tmp)
            probes.extend(await _probe_live(server_module))
            probes.extend(await _probe_live_retrieval_robustness(server_module))
            probes.extend(await _probe_live_latency_and_cost(server_module))

        by_category: dict[str, dict[str, Any]] = {}
        for probe in probes:
            bucket = by_category.setdefault(probe.category, {"score": 0, "max_score": 0, "probes": []})
            if probe.passed:
                bucket["score"] += probe.points
            bucket["max_score"] += probe.points
            bucket["probes"].append(asdict(probe))

        total_score = sum(item["score"] for item in by_category.values())
        total_max_score = sum(item["max_score"] for item in by_category.values())
        gating_failures = _failed_gating_probes(by_category)

        return {
            "repo_path": str(repo_path),
            "total_score": total_score,
            "total_max_score": total_max_score,
            "categories": by_category,
            "gating_failures": gating_failures,
        }
    finally:
        _restore_env(env_snapshot)
        shutil.rmtree(score_tmp, ignore_errors=True)


def _archive_baseline(repo_root: Path, baseline_ref: str) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="grok-baseline-"))
    result = subprocess.run(
        ["git", "-C", str(repo_root), "archive", baseline_ref],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
        try:
            archive.extractall(temp_dir, filter="data")
        except TypeError:
            archive.extractall(temp_dir)
    return temp_dir


def _build_subprocess_pythonpath(repo_path: Path) -> str:
    paths: list[str] = [str(repo_path / "src")]

    for scheme in ("purelib", "platlib"):
        candidate = sysconfig.get_path(scheme)
        if candidate:
            paths.append(candidate)

    for entry in sys.path:
        if not entry:
            continue
        if "__editable__" in entry:
            continue
        if "site-packages" in entry or "dist-packages" in entry:
            paths.append(entry)

    deduped: list[str] = []
    for entry in paths:
        if entry and entry not in deduped:
            deduped.append(entry)
    return os.pathsep.join(deduped)


def _run_score_subprocess(repo_path: Path, include_live: bool) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-S",
        str(Path(__file__).resolve()),
        "--score-repo",
        str(repo_path),
    ]
    if include_live:
        cmd.append("--include-live")

    env = os.environ.copy()
    env["PYTHONPATH"] = _build_subprocess_pythonpath(repo_path)
    env["PYTHONNOUSERSITE"] = "1"

    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            env=env,
            cwd=repo_path,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to score repo {repo_path}\nstdout:\n{exc.stdout}\nstderr:\n{exc.stderr}"
        ) from exc

    return json.loads(completed.stdout)


def compare_scores(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    candidate_categories = candidate["categories"]
    baseline_categories = baseline["categories"]
    category_names = sorted(set(candidate_categories) | set(baseline_categories))

    deltas: dict[str, Any] = {}
    all_non_decreasing = True
    for name in category_names:
        current_score = candidate_categories.get(name, {}).get("score", 0)
        baseline_score = baseline_categories.get(name, {}).get("score", 0)
        delta = current_score - baseline_score
        deltas[name] = {
            "candidate": current_score,
            "baseline": baseline_score,
            "delta": delta,
        }
        if delta < 0:
            all_non_decreasing = False

    total_delta = candidate["total_score"] - baseline["total_score"]
    candidate_failed_gating = set(candidate.get("gating_failures", []))
    baseline_failed_gating = set(baseline.get("gating_failures", []))
    new_failed_gating = sorted(candidate_failed_gating - baseline_failed_gating)
    remaining_failed_gating = sorted(candidate_failed_gating)
    accepted = total_delta > 0 and all_non_decreasing and not new_failed_gating

    return {
        "accepted": accepted,
        "total": {
            "candidate": candidate["total_score"],
            "baseline": baseline["total_score"],
            "delta": total_delta,
        },
        "categories": deltas,
        "new_failed_gating_probes": new_failed_gating,
        "remaining_failed_gating_probes": remaining_failed_gating,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare current grok-search against the original GitHub baseline.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--baseline-ref", default=DEFAULT_BASELINE_REF)
    parser.add_argument("--include-live", action="store_true")
    parser.add_argument("--score-repo", help="Internal mode: score a single repo path and emit JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.score_repo:
        result = asyncio.run(score_repo(Path(args.score_repo), args.include_live))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    repo_root = Path(args.repo_root).resolve()
    baseline_path = _archive_baseline(repo_root, args.baseline_ref)
    try:
        candidate = _run_score_subprocess(repo_root, args.include_live)
        baseline = _run_score_subprocess(baseline_path, args.include_live)
        comparison = compare_scores(candidate, baseline)
        payload = {
            "baseline_ref": args.baseline_ref,
            "candidate": candidate,
            "baseline": baseline,
            "comparison": comparison,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        shutil.rmtree(baseline_path, ignore_errors=True)


if __name__ == "__main__":
    main()
