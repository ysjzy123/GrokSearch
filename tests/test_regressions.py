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
from grok_search.fetch_processing import augment_fetched_markdown, extract_html_tables, infer_title
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
    monkeypatch.delenv("EXA_API_KEY", raising=False)
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


def test_config_normalizes_shell_quoted_env_values(monkeypatch):
    monkeypatch.setenv("GROK_API_URL", "'https://api.example.test/v1'")
    monkeypatch.setenv("GROK_API_KEY", "'test-key'")
    monkeypatch.setenv("TAVILY_API_URL", "\"http://127.0.0.1:8080\"")
    monkeypatch.setenv("TAVILY_ENABLED", "'true'")

    server.config._cached_model = None

    assert server.config.grok_api_url == "https://api.example.test/v1"
    assert server.config.grok_api_key == "test-key"
    assert server.config.tavily_api_url == "http://127.0.0.1:8080"
    assert server.config.tavily_enabled is True


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


def test_augment_fetched_markdown_adds_source_metadata_and_original_content():
    content = "# Shanghai Gold Exchange\n\n| 日期 | 合约 | 最高价 |\n| --- | --- | --- |\n| 2026-01-29 | Au99.99 | 1256.00 |"

    result = augment_fetched_markdown("https://example.com/sge", content)

    assert result.startswith("---\nsource_url: https://example.com/sge")
    assert "inferred_title: Shanghai Gold Exchange" in result
    assert "## Extraction Aids" in result
    assert "### Normalized Tables" in result
    assert "## Original Content" in result
    assert content in result


def test_augment_fetched_markdown_normalizes_reddit_json_payload():
    content = json.dumps(
        [
            {
                "kind": "Listing",
                "data": {
                    "children": [
                        {
                            "kind": "t3",
                            "data": {
                                "title": "What are you building with Python this week?",
                                "subreddit_name_prefixed": "r/Python",
                                "author": "weekly-bot",
                                "selftext": "Share your projects and progress.",
                                "score": 42,
                                "num_comments": 9,
                                "permalink": "/r/Python/comments/1c5qg8q/what_are_you_building_with_python_this_week/",
                            },
                        }
                    ]
                },
            },
            {"kind": "Listing", "data": {"children": []}},
        ],
        ensure_ascii=False,
    )

    result = augment_fetched_markdown("https://www.reddit.com/comments/1c5qg8q/.json?raw_json=1", content)

    assert "inferred_title: What are you building with Python this week?" in result
    assert "source_format: reddit_json" in result
    assert "subreddit: r/Python" in result
    assert "Share your projects and progress." in result
    assert '"kind": "Listing"' not in result


def test_infer_title_skips_yaml_metadata_fields():
    content = "\n".join([
        "source_url: https://github.com/lsdefine/GenericAgent/tree/main",
        "inferred_title: main",
        "",
        "# lsdefine/GenericAgent",
        "",
        "Repository body",
    ])

    assert infer_title(content, "https://github.com/lsdefine/GenericAgent/tree/main") == "lsdefine/GenericAgent"


def test_infer_title_prefers_github_heading_over_shell_text():
    content = "\n".join([
        "Skip to content",
        "Search or jump to...",
        "",
        "# lsdefine/GenericAgent",
        "",
        "Repository body",
    ])

    assert infer_title(content, "https://github.com/lsdefine/GenericAgent/tree/main") == "lsdefine/GenericAgent"


def test_extract_html_tables_normalizes_header_and_rows():
    html = (
        "<table><tr><th>日期</th><th>合约</th><th>最高价</th></tr>"
        "<tr><td>2026-01-29</td><td>Au99.99</td><td>1256.00</td></tr></table>"
    )

    tables = extract_html_tables(html)

    assert len(tables) == 1
    assert "| 日期 | 合约 | 最高价 |" in tables[0]
    assert "| 2026-01-29 | Au99.99 | 1256.00 |" in tables[0]


@pytest.mark.asyncio
async def test_web_fetch_wraps_result_with_extraction_aids(monkeypatch):
    monkeypatch.setattr(server, "_call_tavily_extract", lambda _url: asyncio.sleep(0, result="# Example\n\n| A | B |\n| --- | --- |\n| 1 | 2 |"))
    monkeypatch.setattr(server, "_call_firecrawl_scrape", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_exa_contents", lambda _url, _ctx=None: asyncio.sleep(0, result=None))

    result = await server.web_fetch("https://example.com/page")

    assert "source_url: https://example.com/page" in result
    assert "## Extraction Aids" in result
    assert "### Normalized Tables" in result
    assert "## Original Content" in result


@pytest.mark.asyncio
async def test_web_fetch_uses_exa_when_other_extractors_fail(monkeypatch):
    monkeypatch.setattr(server, "_call_tavily_extract", lambda _url: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_firecrawl_scrape", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_exa_contents", lambda _url, _ctx=None: asyncio.sleep(0, result="# Exa Title\n\nBody"))

    result = await server.web_fetch("https://example.com/page")

    assert "source_url: https://example.com/page" in result
    assert "## Original Content" in result
    assert "# Exa Title" in result


@pytest.mark.asyncio
async def test_web_fetch_falls_back_to_exa_when_tavily_result_is_low_quality(monkeypatch):
    monkeypatch.setattr(server, "_call_tavily_extract", lambda _url: asyncio.sleep(0, result="# Navigation Menu\n\nSaved searches\n\nProvide feedback"))
    monkeypatch.setattr(server, "_call_firecrawl_scrape", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_exa_contents", lambda _url, _ctx=None: asyncio.sleep(0, result="# Actual Issue Title\n\nUseful discussion body"))

    result = await server.web_fetch("https://example.com/page")

    assert "Actual Issue Title" in result
    assert "Navigation Menu" not in result


@pytest.mark.asyncio
async def test_web_fetch_selects_richer_firecrawl_candidate_over_short_tavily(monkeypatch):
    tavily_content = "# Short Note\n\nBrief summary."
    firecrawl_content = "\n".join([
        "# Deep Dive",
        "",
        "## Background",
        "This candidate contains a longer and more substantive explanation of the page body, including details that should make it win the selector.",
        "",
        "## Steps",
        "1. First observe the live behavior.",
        "2. Then compare the extracted output.",
        "3. Finally validate the root cause with a clean reproduction.",
        "",
        "```text",
        "important trace details",
        "```",
    ])

    monkeypatch.setattr(server, "_call_tavily_extract", lambda _url: asyncio.sleep(0, result=tavily_content))
    monkeypatch.setattr(server, "_call_firecrawl_scrape", lambda _url, _ctx=None: asyncio.sleep(0, result=firecrawl_content))
    monkeypatch.setattr(server, "_call_exa_contents", lambda _url, _ctx=None: asyncio.sleep(0, result=None))

    result = await server.web_fetch("https://example.com/page")

    assert "Deep Dive" in result
    assert "Brief summary." not in result


@pytest.mark.asyncio
async def test_web_fetch_uses_juejin_metadata_fallback_when_extractors_fail(monkeypatch):
    monkeypatch.setattr(server, "_call_tavily_extract", lambda _url: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_firecrawl_scrape", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_exa_contents", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(
        server,
        "_fetch_raw_html",
        lambda _url, _ctx=None: asyncio.sleep(
            0,
            result=(
                "<html><head>"
                "<title>Juejin Live Article - 掘金</title>"
                '<meta name="description" content="This article explains how to automate publishing across platforms.">'
                "</head></html>"
            ),
        ),
    )

    result = await server.web_fetch("https://juejin.cn/post/7629183326780276763")

    assert "Juejin Live Article - 掘金" in result
    assert "fallback_mode: metadata_summary" in result
    assert "页面正文受限，回退为页面元信息摘要。" in result


@pytest.mark.asyncio
async def test_web_fetch_uses_github_metadata_fallback_when_extractors_fail(monkeypatch):
    monkeypatch.setattr(server, "_call_tavily_extract", lambda _url: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_firecrawl_scrape", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_exa_contents", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(
        server,
        "_fetch_raw_html",
        lambda _url, _ctx=None: asyncio.sleep(
            0,
            result=(
                "<html><head>"
                "<title>GitHub - lsdefine/GenericAgent: Self-evolving agent · GitHub</title>"
                '<meta name="description" content="Self-evolving agent: grows skill tree from 3.3K-line seed.">'
                "</head></html>"
            ),
        ),
    )

    result = await server.web_fetch("https://github.com/lsdefine/GenericAgent/tree/main")

    assert "GitHub - lsdefine/GenericAgent: Self-evolving agent · GitHub" in result
    assert "Self-evolving agent: grows skill tree from 3.3K-line seed." in result
    assert "GitHub 页面启用站点级元信息回退。" in result


@pytest.mark.asyncio
async def test_firecrawl_scrape_skips_github_urls(monkeypatch):
    import httpx

    monkeypatch.setenv("FIRECRAWL_API_KEY", "f-key")

    class FailIfConstructed:
        def __init__(self, *args, **kwargs):
            raise AssertionError("httpx.AsyncClient should not be constructed for GitHub scrape")

    monkeypatch.setattr(httpx, "AsyncClient", FailIfConstructed)

    result = await server._call_firecrawl_scrape("https://github.com/lsdefine/GenericAgent/tree/main")

    assert result is None


@pytest.mark.asyncio
async def test_web_fetch_uses_reddit_search_fallback_when_page_is_blocked(monkeypatch):
    monkeypatch.setattr(server, "_call_tavily_extract", lambda _url: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_firecrawl_scrape", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_exa_contents", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_fetch_reddit_json_fallback", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(
        server,
        "_fetch_raw_html",
        lambda _url, _ctx=None: asyncio.sleep(0, result="<html><body>You've been blocked by network security.</body></html>"),
    )
    monkeypatch.setattr(
        server,
        "_call_tavily_search",
        lambda _query, max_results=3: asyncio.sleep(
            0,
            result=[
                {
                    "title": "What are you building with Python this week?",
                    "url": "https://www.reddit.com/r/Python/comments/1c5qg8q/what_are_you_building_with_python_this_week/",
                    "content": "Weekly discussion thread for Python builders sharing projects and progress.",
                }
            ],
        ),
    )
    monkeypatch.setattr(server, "_call_exa_search", lambda _query, max_results=3, ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_firecrawl_search", lambda _query, limit=3: asyncio.sleep(0, result=None))

    result = await server.web_fetch("https://www.reddit.com/r/Python/comments/1c5qg8q/what_are_you_building_with_python_this_week/")

    assert "What are you building with Python this week?" in result
    assert "Weekly discussion thread for Python builders" in result
    assert "目标页被反爬拦截，回退为搜索结果摘要。" in result


@pytest.mark.asyncio
async def test_web_fetch_uses_zhihu_search_fallback_when_page_is_blocked(monkeypatch):
    monkeypatch.setattr(server, "_call_tavily_extract", lambda _url: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_firecrawl_scrape", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_exa_contents", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_fetch_reddit_json_fallback", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(
        server,
        "_fetch_raw_html",
        lambda _url, _ctx=None: asyncio.sleep(
            0,
            result="<html><body>知乎，让每一次点击都充满意义 —— 欢迎来到知乎，发现问题背后的世界。</body></html>",
        ),
    )
    monkeypatch.setattr(
        server,
        "_call_tavily_search",
        lambda _query, max_results=3: asyncio.sleep(
            0,
            result=[
                {
                    "title": "如何评价某个问题的最佳实践 - 知乎",
                    "url": "https://www.zhihu.com/question/649365581",
                    "content": "该问题讨论了实践方式、取舍以及常见误区。",
                }
            ],
        ),
    )
    monkeypatch.setattr(server, "_call_exa_search", lambda _query, max_results=3, ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_firecrawl_search", lambda _query, limit=3: asyncio.sleep(0, result=None))

    result = await server.web_fetch("https://www.zhihu.com/question/649365581")

    assert "如何评价某个问题的最佳实践 - 知乎" in result
    assert "该问题讨论了实践方式、取舍以及常见误区。" in result
    assert "目标页被知乎风控拦截，回退为搜索结果摘要。" in result


@pytest.mark.asyncio
async def test_web_fetch_prefers_exa_search_recovery_when_available(monkeypatch):
    monkeypatch.setattr(server, "_call_tavily_extract", lambda _url: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_firecrawl_scrape", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_exa_contents", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_fetch_reddit_json_fallback", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(
        server,
        "_fetch_raw_html",
        lambda _url, _ctx=None: asyncio.sleep(0, result="<html><body>You've been blocked by network security.</body></html>"),
    )
    monkeypatch.setattr(
        server,
        "_call_exa_search",
        lambda _query, max_results=3, ctx=None: asyncio.sleep(
            0,
            result=[
                {
                    "provider": "exa_search",
                    "url": "https://www.reddit.com/r/Python/comments/1c5qg8q/what_are_you_building_with_python_this_week/",
                    "title": "Weekly Python Thread",
                    "content": "Exa recovered a stronger summary for the blocked Reddit thread.",
                    "score": 0.9,
                }
            ],
        ),
    )
    monkeypatch.setattr(server, "_call_tavily_search", lambda _query, max_results=3: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_firecrawl_search", lambda _query, limit=3: asyncio.sleep(0, result=None))

    result = await server.web_fetch("https://www.reddit.com/r/Python/comments/1c5qg8q/what_are_you_building_with_python_this_week/")

    assert "Weekly Python Thread" in result
    assert "Exa recovered a stronger summary" in result


def test_is_reliable_zhihu_search_recovery_result_filters_generic_pages():
    generic = {
        "url": "https://www.zhihu.com/app/",
        "title": "知乎客户端",
        "content": "请您登录后查看更多专业优质内容。",
    }
    exact = {
        "url": "https://www.zhihu.com/question/649365581",
        "title": "如何评价某个问题的最佳实践 - 知乎",
        "content": "该问题讨论了实践方式、取舍以及常见误区。",
    }

    assert server._is_reliable_zhihu_search_recovery_result(generic, "https://www.zhihu.com/question/649365581") is False
    assert server._is_reliable_zhihu_search_recovery_result(exact, "https://www.zhihu.com/question/649365581") is True


def test_is_reliable_reddit_search_recovery_result_prefers_matching_post_id():
    target_url = "https://www.reddit.com/r/Python/comments/1c5qg8q/what_are_you_building_with_python_this_week/"
    unrelated = {
        "url": "https://www.reddit.com/r/vandwellers/comments/1xyz987/best_portable_power_station_for_the_lowest_price/",
        "title": "Best portable power station for the lowest price?",
        "content": "Completely unrelated Reddit thread.",
    }
    exact = {
        "url": target_url,
        "title": "What are you building with Python this week?",
        "content": "Weekly thread for sharing Python projects.",
    }

    assert server._is_reliable_reddit_search_recovery_result(unrelated, target_url) is False
    assert server._is_reliable_reddit_search_recovery_result(exact, target_url) is True


def test_select_best_search_recovery_result_uses_reddit_specific_matching():
    target_url = "https://www.reddit.com/r/programming/comments/1b5m4s6/what_happened_to_stack_overflow/"
    wrong = {
        "provider": "tavily_search",
        "url": "https://www.reddit.com/r/ExperiencedDevs/comments/abcd123/was_it_ai_or_did_the_platform_kill_itself_with_elitism/",
        "title": "Was it AI, or did the platform kill itself with elitism?",
        "content": "This is a different discussion with longer content that should be filtered out.",
        "score": 0.95,
    }
    right = {
        "provider": "exa_search",
        "url": target_url,
        "title": "What happened to Stack Overflow?",
        "content": "Discussion about Stack Overflow's decline and community changes.",
        "score": 0.2,
    }

    winner = server._select_best_search_recovery_result([wrong, right], target_url)

    assert winner == right


@pytest.mark.asyncio
async def test_web_fetch_uses_reddit_identity_fallback_when_search_recovery_has_no_reliable_results(monkeypatch):
    monkeypatch.setattr(server, "_call_tavily_extract", lambda _url: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_firecrawl_scrape", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_exa_contents", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_fetch_reddit_json_fallback", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(
        server,
        "_fetch_raw_html",
        lambda _url, _ctx=None: asyncio.sleep(0, result="<html><body>You've been blocked by network security.</body></html>"),
    )
    monkeypatch.setattr(server, "_call_exa_search", lambda _query, max_results=3, ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(
        server,
        "_call_tavily_search",
        lambda _query, max_results=3: asyncio.sleep(
            0,
            result=[
                {
                    "title": "Best portable power station for the lowest price?",
                    "url": "https://www.reddit.com/r/vandwellers/comments/1xyz987/best_portable_power_station_for_the_lowest_price/",
                    "content": "Wrong Reddit thread returned by search.",
                }
            ],
        ),
    )
    monkeypatch.setattr(server, "_call_firecrawl_search", lambda _query, limit=3: asyncio.sleep(0, result=None))

    result = await server.web_fetch("https://www.reddit.com/r/Python/comments/1c5qg8q/what_are_you_building_with_python_this_week/")

    assert "Reddit r/Python thread 1c5qg8q (URL-derived topic: what are you building with python this week)" in result
    assert "fallback_mode: url_identity_summary" in result
    assert "different Reddit post" in result
    assert "not treated as a verified public title" in result


@pytest.mark.asyncio
async def test_web_fetch_uses_reddit_identity_fallback_when_raw_html_fetch_fails(monkeypatch):
    monkeypatch.setattr(server, "_call_tavily_extract", lambda _url: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_firecrawl_scrape", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_exa_contents", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_fetch_reddit_json_fallback", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_fetch_raw_html", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_exa_search", lambda _query, max_results=3, ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_tavily_search", lambda _query, max_results=3: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_firecrawl_search", lambda _query, limit=3: asyncio.sleep(0, result=None))

    result = await server.web_fetch("https://www.reddit.com/r/Python/comments/1c5qg8q/what_are_you_building_with_python_this_week/")

    assert "Reddit r/Python thread 1c5qg8q (URL-derived topic: what are you building with python this week)" in result
    assert "fallback_mode: url_identity_summary" in result
    assert "目标页抓取失败；未恢复到同帖公开摘要" in result


@pytest.mark.asyncio
async def test_web_fetch_prefers_reddit_json_fallback_when_available(monkeypatch):
    reddit_json = json.dumps(
        [
            {
                "kind": "Listing",
                "data": {
                    "children": [
                        {
                            "kind": "t3",
                            "data": {
                                "title": "What are you building with Python this week?",
                                "subreddit_name_prefixed": "r/Python",
                                "author": "weekly-bot",
                                "selftext": "Share your projects and progress.",
                                "score": 42,
                                "num_comments": 9,
                                "permalink": "/r/Python/comments/1c5qg8q/what_are_you_building_with_python_this_week/",
                            },
                        }
                    ]
                },
            },
            {"kind": "Listing", "data": {"children": []}},
        ],
        ensure_ascii=False,
    )

    monkeypatch.setattr(server, "_call_tavily_extract", lambda _url: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_firecrawl_scrape", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_exa_contents", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_fetch_reddit_json_fallback", lambda _url, _ctx=None: asyncio.sleep(0, result=reddit_json))
    monkeypatch.setattr(
        server,
        "_fetch_raw_html",
        lambda _url, _ctx=None: asyncio.sleep(0, result="<html><body>You've been blocked by network security.</body></html>"),
    )
    monkeypatch.setattr(server, "_call_exa_search", lambda _query, max_results=3, ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_tavily_search", lambda _query, max_results=3: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_firecrawl_search", lambda _query, limit=3: asyncio.sleep(0, result=None))

    result = await server.web_fetch("https://www.reddit.com/r/Python/comments/1c5qg8q/what_are_you_building_with_python_this_week/")

    assert "What are you building with Python this week?" in result
    assert "Share your projects and progress." in result
    assert "fallback_mode: reddit_json_summary" in result
    assert "json_recovery_provider:" in result
    assert "目标页被 Reddit 反爬拦截，回退为同帖 JSON 元数据摘要。" in result


@pytest.mark.asyncio
async def test_web_fetch_uses_zhihu_identity_fallback_when_search_recovery_has_no_reliable_results(monkeypatch):
    monkeypatch.setattr(server, "_call_tavily_extract", lambda _url: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_firecrawl_scrape", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_exa_contents", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(
        server,
        "_fetch_raw_html",
        lambda _url, _ctx=None: asyncio.sleep(
            0,
            result="<html><body>知乎，让每一次点击都充满意义 —— 欢迎来到知乎，发现问题背后的世界。</body></html>",
        ),
    )
    monkeypatch.setattr(server, "_call_exa_search", lambda _query, max_results=3, ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(
        server,
        "_call_tavily_search",
        lambda _query, max_results=3: asyncio.sleep(
            0,
            result=[
                {
                    "title": "知乎客户端",
                    "url": "https://www.zhihu.com/app/",
                    "content": "请您登录后查看更多专业优质内容。",
                }
            ],
        ),
    )
    monkeypatch.setattr(server, "_call_firecrawl_search", lambda _query, limit=3: asyncio.sleep(0, result=None))

    result = await server.web_fetch("https://www.zhihu.com/question/649365581")

    assert "知乎问题 649365581" in result
    assert "fallback_mode: url_identity_summary" in result
    assert "可验证的公开标题或摘要" in result


@pytest.mark.asyncio
async def test_build_zhihu_search_recovery_queries_covers_question_and_article_ids():
    question_queries = server._build_zhihu_search_recovery_queries(
        "https://www.zhihu.com/question/649365581/answer/3552952553"
    )
    article_queries = server._build_zhihu_search_recovery_queries(
        "https://zhuanlan.zhihu.com/p/638427200"
    )

    assert question_queries == (
        "site:zhihu.com https://www.zhihu.com/question/649365581/answer/3552952553",
        "site:zhihu.com/question/649365581",
        "site:zhihu.com/question 649365581",
        "site:zhihu.com/question/649365581/answer/3552952553",
        "site:zhihu.com/answer 3552952553",
    )
    assert article_queries == (
        "site:zhuanlan.zhihu.com https://zhuanlan.zhihu.com/p/638427200",
        "site:zhuanlan.zhihu.com/p/638427200",
        "site:zhuanlan.zhihu.com 638427200",
    )


@pytest.mark.asyncio
async def test_zhihu_search_recovery_filters_shell_results_from_title_and_url(monkeypatch):
    shell_like = {
        "url": "https://www.zhihu.com/question/649365581?utm_psn=bad",
        "title": "知乎，让每一次点击都充满意义",
        "content": "欢迎来到知乎，发现问题背后的世界 need_login=true",
    }
    reliable = {
        "url": "https://www.zhihu.com/question/649365581",
        "title": "如何评价某个问题？ - 知乎",
        "content": "这是该知乎问题的公开摘要片段。",
    }

    assert server._is_reliable_zhihu_search_recovery_result(
        shell_like,
        "https://www.zhihu.com/question/649365581",
    ) is False
    assert server._is_reliable_zhihu_search_recovery_result(
        reliable,
        "https://www.zhihu.com/question/649365581",
    ) is True


@pytest.mark.asyncio
async def test_web_fetch_returns_failure_when_site_fallback_recovery_has_no_results(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "dummy-key")
    monkeypatch.setattr(server, "_call_tavily_extract", lambda _url: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_firecrawl_scrape", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_exa_contents", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(
        server,
        "_fetch_raw_html",
        lambda _url, _ctx=None: asyncio.sleep(
            0,
            result="<html><body>知乎，让每一次点击都充满意义 —— 欢迎来到知乎，发现问题背后的世界。</body></html>",
        ),
    )
    monkeypatch.setattr(server, "_call_exa_search", lambda _query, max_results=3, ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_tavily_search", lambda _query, max_results=3: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_firecrawl_search", lambda _query, limit=3: asyncio.sleep(0, result=None))

    result = await server.web_fetch("https://www.zhihu.com/question/649365581")

    assert "知乎问题 649365581" in result
    assert "fallback_mode: url_identity_summary" in result


def test_select_best_fetch_candidate_prefers_non_low_quality_candidate():
    low_quality = server._build_fetch_candidate(
        "exa",
        "\n".join([
            "Navigation Menu",
            "Skip to content",
            "Saved searches",
            "Provide feedback",
            "Sign in",
        ]),
        "https://github.com/org/repo/issues/1",
    )
    valid = server._build_fetch_candidate(
        "tavily",
        "\n".join([
            "# Actual Issue",
            "",
            "## Details",
            "This is the real issue body with meaningful text that should outrank the shell content.",
            "",
            "## Notes",
            "The response includes enough structure to be considered substantive.",
        ]),
        "https://github.com/org/repo/issues/1",
    )

    winner = server._select_best_fetch_candidate([low_quality, valid])

    assert winner is not None
    assert winner["provider"] == "tavily"
    assert winner["is_low_quality"] is False


def test_low_quality_fetch_detection_flags_github_shell_content():
    content = "\n".join([
        "Navigation Menu",
        "Skip to content",
        "Search or jump to...",
        "Saved searches",
        "Use saved searches to filter your results more quickly",
        "Sign in",
    ])

    assert server._is_low_quality_fetch_result(content, "https://github.com/org/repo/issues/123") is True


def test_low_quality_fetch_detection_flags_reddit_app_shell_content():
    content = "\n".join([
        "Reddit - Dive into anything",
        "Open menu",
        "Create account",
        "Get App",
        "Back to Top",
        "Use App",
    ])

    assert server._is_low_quality_fetch_result(content, "https://www.reddit.com/r/python/comments/abc123/example") is True


def test_low_quality_fetch_detection_flags_reddit_verification_page():
    content = "\n".join([
        "Reddit - Please wait for verification",
        "js_challenge",
        "solution",
    ])

    assert server._is_low_quality_fetch_result(content, "https://www.reddit.com/r/python/comments/abc123/example") is True


def test_low_quality_fetch_detection_flags_zhihu_login_shell_content():
    content = "\n".join([
        "知乎，让每一次点击都充满意义",
        "打开知乎 App",
        "登录/注册后即可查看更多内容",
        "查看全部回答",
        "写回答",
    ])

    assert server._is_low_quality_fetch_result(content, "https://www.zhihu.com/question/123456") is True


def test_low_quality_fetch_detection_flags_zhihu_risk_page():
    content = "\n".join([
        "知乎，让每一次点击都充满意义 —— 欢迎来到知乎，发现问题背后的世界。",
        "account/unhuman?need_login=true",
        "zse-ck",
    ])

    assert server._is_low_quality_fetch_result(content, "https://www.zhihu.com/question/123456") is True


def test_low_quality_fetch_detection_flags_juejin_login_shell_content():
    content = "\n".join([
        "稀土掘金",
        "打开 App",
        "登录后查看更多优质内容",
        "点赞",
        "评论",
        "收藏",
    ])

    assert server._is_low_quality_fetch_result(content, "https://juejin.cn/post/123456789") is True


def test_low_quality_fetch_detection_flags_juejin_not_found_shell_content():
    content = "\n".join([
        "找不到页面",
        "\"statusCode\":404",
        "\"errorView\":\"NotFoundView\"",
        "verifyCenter",
    ])

    assert server._is_low_quality_fetch_result(content, "https://juejin.cn/post/123456789") is True


def test_low_quality_fetch_detection_flags_cnblogs_sidebar_shell_content():
    content = "\n".join([
        "公告",
        "昵称：test",
        "园龄：3年",
        "粉丝：10",
        "关注：2",
        "积分与排名",
        "随笔档案",
        "阅读排行榜",
    ])

    assert server._is_low_quality_fetch_result(content, "https://www.cnblogs.com/test/p/example.html") is True


def test_low_quality_fetch_detection_flags_cnblogs_404_page():
    content = "\n".join([
        "[![](https://common.cnblogs.com/logo.svg)](https://www.cnblogs.com/)",
        "404 - 您访问的页面不存在",
        "可能是网址有误，或者对应的内容已被删除，或者处于私有状态",
        "邮件联系：contact@cnblogs.com",
    ])

    assert server._is_low_quality_fetch_result(content, "https://www.cnblogs.com/test/p/missing.html") is True


def test_low_quality_fetch_detection_keeps_real_github_issue_content():
    content = "\n".join([
        "# Bug: MCP web_fetch returns shell page",
        "",
        "## Description",
        "When requesting a GitHub Issue URL through the fetch pipeline, the response sometimes returns only navigation chrome instead of the actual discussion body.",
        "",
        "## Reproduction",
        "1. Open the issue URL directly.",
        "2. Observe that the page contains the full issue body and comments.",
        "3. Compare it with the extracted markdown output.",
        "",
        "```text",
        "Expected: full issue content",
        "Actual: shell content only",
        "```",
    ])

    assert server._is_low_quality_fetch_result(content, "https://github.com/org/repo/issues/123") is False


def test_build_fetch_candidate_penalizes_firecrawl_for_github_shell_pages():
    exa_candidate = server._build_fetch_candidate(
        "exa",
        "\n".join([
            "# lsdefine/GenericAgent",
            "",
            "## Overview",
            "Self-evolving agent that grows a reusable skill tree.",
        ]),
        "https://github.com/lsdefine/GenericAgent/tree/main",
    )
    firecrawl_candidate = server._build_fetch_candidate(
        "firecrawl",
        "\n".join([
            "# lsdefine/GenericAgent",
            "",
            "## Overview",
            "Self-evolving agent that grows a reusable skill tree.",
        ]),
        "https://github.com/lsdefine/GenericAgent/tree/main",
    )

    assert exa_candidate is not None
    assert firecrawl_candidate is not None
    assert exa_candidate["score"] > firecrawl_candidate["score"]


@pytest.mark.asyncio
async def test_web_fetch_reports_missing_all_fetch_keys(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setattr(server, "_call_tavily_extract", lambda _url: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_firecrawl_scrape", lambda _url, _ctx=None: asyncio.sleep(0, result=None))
    monkeypatch.setattr(server, "_call_exa_contents", lambda _url, _ctx=None: asyncio.sleep(0, result=None))

    result = await server.web_fetch("https://example.com/page")

    assert "TAVILY_API_KEY" in result
    assert "FIRECRAWL_API_KEY" in result
    assert "EXA_API_KEY" in result


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
    assert calls["tavily"] == ("capital of france", 3)
    assert calls["firecrawl"] == ("capital of france", 3)
    assert calls["describe"] == ["https://t.example/item"]
    assert "rank" not in calls
    by_provider = {item["provider"]: item for item in sources["sources"]}
    assert by_provider["tavily"]["title"] == "Enriched Tavily"
    assert by_provider["tavily"]["description"] == "Enriched summary"
    assert by_provider["firecrawl"]["provider"] == "firecrawl"


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
    assert "overview" in facets or "runtime" in facets or "best-practices" in facets
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
            {"url": "https://ownership.example/guide", "title": "Rust Ownership", "content": "ownership memory safety", "facet": "ownership"},
            {"url": "https://roadmap.example/path", "title": "Rust Roadmap", "content": "learning roadmap", "facet": "roadmap"},
            {"url": "https://book.example/intro", "title": "Rust Book", "content": "official guide and examples", "facet": "overview"},
        ]

    async def fake_firecrawl(query, limit=14):
        calls.append(("firecrawl", query, limit))
        return [
            {"title": "Tokio Runtime Guide", "url": "https://tokio.example/runtime", "description": "tokio async runtime network services", "facet": "runtime"},
            {"title": "Rust Patterns", "url": "https://patterns.example/rust", "description": "best practices and design patterns", "facet": "best-practices"},
            {"title": "Cargo Tooling", "url": "https://cargo.example/tools", "description": "cargo tooling workflow", "facet": "tooling"},
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
    assert summary["budget_unused"] > 0
    assert phases["expansion"]["skipped"] is True
    assert phases["expansion"]["reason"] == "initial_support_sufficient"
    assert len(calls) == 2


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

    ranked_sources, metadata = await server._enrich_and_rank_sources("example query", SlowProvider(), None, sources)

    assert ranked_sources == sources
    assert metadata["enrichment_applied"] is False
    assert metadata["ranking_applied"] is False
