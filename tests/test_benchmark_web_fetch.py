import importlib.util
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_web_fetch.py"


def load_benchmark_web_fetch():
    module_name = "benchmark_web_fetch_test_module"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_live_env_from_codex_config_reads_grok_search_env(tmp_path):
    benchmark = load_benchmark_web_fetch()

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[mcp_servers.grok-search.env]
GROK_API_URL = "https://example.test/v1"
GROK_API_KEY = "grok-key"
TAVILY_API_KEY = "t-key"
FIRECRAWL_API_KEY = "f-key"
EXA_API_KEY = "e-key"
EXA_ENABLED = true
""".strip(),
        encoding="utf-8",
    )

    loaded = benchmark._load_live_env_from_codex_config(config_path)

    assert loaded["GROK_API_URL"] == "https://example.test/v1"
    assert loaded["GROK_API_KEY"] == "grok-key"
    assert loaded["TAVILY_API_KEY"] == "t-key"
    assert loaded["FIRECRAWL_API_KEY"] == "f-key"
    assert loaded["EXA_API_KEY"] == "e-key"
    assert loaded["EXA_ENABLED"] == "true"


def test_apply_live_env_from_codex_config_strips_proxy_and_respects_prefer_existing(monkeypatch, tmp_path):
    benchmark = load_benchmark_web_fetch()

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[mcp_servers.grok-search.env]
GROK_API_URL = "https://example.test/v1"
GROK_API_KEY = "config-key"
EXA_API_KEY = "config-exa"
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("GROK_API_KEY", "existing-key")
    monkeypatch.delenv("EXA_API_KEY", raising=False)

    _, loaded_keys = benchmark._apply_live_env_from_codex_config(config_path, prefer_existing=True)

    assert "HTTP_PROXY" not in os.environ
    assert "HTTPS_PROXY" not in os.environ
    assert os.environ["GROK_API_URL"] == "https://example.test/v1"
    assert os.environ["GROK_API_KEY"] == "existing-key"
    assert os.environ["EXA_API_KEY"] == "config-exa"
    assert loaded_keys == ["EXA_API_KEY", "GROK_API_KEY", "GROK_API_URL"]

    benchmark._apply_live_env_from_codex_config(config_path, prefer_existing=False)
    assert os.environ["GROK_API_KEY"] == "config-key"


def test_build_summary_collects_expected_metrics():
    benchmark = load_benchmark_web_fetch()

    providers = ["tavily", "firecrawl", "exa"]
    results = [
        {
            "url": "https://example.com/a",
            "winner": "exa",
            "winner_score": 98.4,
            "best_low_quality_provider": None,
            "final_fetch_ok": True,
            "final_fetch_length": 1500,
            "final_fetch_preview": "A",
            "final_fetch_metadata": {
                "fallback_mode": "reddit_json_summary",
                "source_format": "reddit_json",
                "json_recovery_provider": "tavily_json",
                "fetch_structure_version": 1,
                "inferred_title": "Thread A",
            },
            "providers": [
                {
                    "provider": "tavily",
                    "status": "ok",
                    "elapsed_ms": 110.0,
                    "score": 52.0,
                    "is_low_quality": True,
                    "content_length": 240,
                },
                {
                    "provider": "firecrawl",
                    "status": "empty_or_failed",
                    "elapsed_ms": 210.0,
                },
                {
                    "provider": "exa",
                    "status": "ok",
                    "elapsed_ms": 95.0,
                    "score": 98.4,
                    "is_low_quality": False,
                    "content_length": 1200,
                },
            ],
        },
        {
            "url": "https://example.com/b",
            "winner": None,
            "winner_score": None,
            "best_low_quality_provider": "tavily",
            "final_fetch_ok": False,
            "final_fetch_length": 0,
            "final_fetch_preview": "提取失败",
            "final_fetch_metadata": {},
            "providers": [
                {
                    "provider": "tavily",
                    "status": "ok",
                    "elapsed_ms": 130.0,
                    "score": 30.0,
                    "is_low_quality": True,
                    "content_length": 200,
                },
                {
                    "provider": "firecrawl",
                    "status": "missing_key",
                    "elapsed_ms": 0.0,
                },
                {
                    "provider": "exa",
                    "status": "ok",
                    "elapsed_ms": 80.0,
                    "score": 25.0,
                    "is_low_quality": True,
                    "content_length": 180,
                },
            ],
        },
    ]

    summary = benchmark._build_summary(results, providers)

    assert summary["checked_urls"] == ["https://example.com/a", "https://example.com/b"]
    assert summary["checked_url_count"] == 2
    assert summary["provider_order"] == providers
    assert summary["successful_urls"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["final_fetch_successful_urls"] == 1
    assert summary["final_fetch_success_rate"] == 0.5
    assert summary["final_fetch_structured_urls"] == 1
    assert summary["final_fetch_structured_rate"] == 0.5
    assert summary["final_fetch_fallback_urls"] == 1
    assert summary["final_fetch_fallback_rate"] == 0.5
    assert summary["final_fetch_fallback_mode_counts"] == {"reddit_json_summary": 1}
    assert summary["final_fetch_source_format_counts"] == {"reddit_json": 1}
    assert summary["final_fetch_recovery_provider_counts"] == {"tavily_json": 1}
    assert summary["all_low_quality_urls"] == 1
    assert summary["all_low_quality_rate"] == 0.5
    assert summary["winner_counts"] == {"exa": 1}

    tavily = summary["provider_summary"]["tavily"]
    firecrawl = summary["provider_summary"]["firecrawl"]
    exa = summary["provider_summary"]["exa"]

    assert tavily == {
        "attempted_urls": 2,
        "successful_urls": 2,
        "success_rate": 1.0,
        "non_low_quality_urls": 0,
        "non_low_quality_rate": 0.0,
        "winner_count": 0,
        "avg_content_length": 220.0,
        "avg_elapsed_s": 0.12,
    }
    assert firecrawl == {
        "attempted_urls": 2,
        "successful_urls": 0,
        "success_rate": 0.0,
        "non_low_quality_urls": 0,
        "non_low_quality_rate": 0.0,
        "winner_count": 0,
        "avg_content_length": 0.0,
        "avg_elapsed_s": 0.1,
    }
    assert exa == {
        "attempted_urls": 2,
        "successful_urls": 2,
        "success_rate": 1.0,
        "non_low_quality_urls": 1,
        "non_low_quality_rate": 0.5,
        "winner_count": 1,
        "avg_content_length": 690.0,
        "avg_elapsed_s": 0.09,
    }

    assert summary["summaries"] == [
        {
            "url": "https://example.com/a",
            "winner": "exa",
            "winner_score": 98.4,
            "best_low_quality_provider": None,
            "final_fetch_ok": True,
            "final_fetch_length": 1500,
            "final_fetch_metadata": {
                "fallback_mode": "reddit_json_summary",
                "source_format": "reddit_json",
                "json_recovery_provider": "tavily_json",
                "fetch_structure_version": 1,
                "inferred_title": "Thread A",
            },
            "attempts": [
                {
                    "provider": "tavily",
                    "status": "ok",
                    "elapsed_s": 0.11,
                    "content_length": 240,
                    "low_quality": True,
                    "score": 52.0,
                },
                {
                    "provider": "firecrawl",
                    "status": "empty_or_failed",
                    "elapsed_s": 0.21,
                    "content_length": 0,
                    "low_quality": None,
                    "score": None,
                },
                {
                    "provider": "exa",
                    "status": "ok",
                    "elapsed_s": 0.1,
                    "content_length": 1200,
                    "low_quality": False,
                    "score": 98.4,
                },
            ],
        },
        {
            "url": "https://example.com/b",
            "winner": None,
            "winner_score": None,
            "best_low_quality_provider": "tavily",
            "final_fetch_ok": False,
            "final_fetch_length": 0,
            "final_fetch_metadata": {},
            "attempts": [
                {
                    "provider": "tavily",
                    "status": "ok",
                    "elapsed_s": 0.13,
                    "content_length": 200,
                    "low_quality": True,
                    "score": 30.0,
                },
                {
                    "provider": "firecrawl",
                    "status": "missing_key",
                    "elapsed_s": 0.0,
                    "content_length": 0,
                    "low_quality": None,
                    "score": None,
                },
                {
                    "provider": "exa",
                    "status": "ok",
                    "elapsed_s": 0.08,
                    "content_length": 180,
                    "low_quality": True,
                    "score": 25.0,
                },
            ],
        },
    ]


def test_extract_final_fetch_metadata_parses_front_matter_and_notes():
    benchmark = load_benchmark_web_fetch()

    text = """---
source_url: https://www.reddit.com/r/Python/comments/abc123/example/
inferred_title: Example Thread
fetch_structure_version: 1
---

## Fetch Notes

- source_url: https://www.reddit.com/r/Python/comments/abc123/example/
- fallback_mode: reddit_json_summary
- source_format: reddit_json
- json_recovery_provider: tavily_json
"""

    metadata = benchmark._extract_final_fetch_metadata(text)

    assert metadata == {
        "source_url": "https://www.reddit.com/r/Python/comments/abc123/example/",
        "inferred_title": "Example Thread",
        "fetch_structure_version": 1,
        "fallback_mode": "reddit_json_summary",
        "source_format": "reddit_json",
        "json_recovery_provider": "tavily_json",
    }


def test_async_main_writes_summary_json(monkeypatch, tmp_path, capsys):
    benchmark = load_benchmark_web_fetch()

    fake_results = [
        {
            "url": "https://example.com/a",
            "winner": "exa",
            "winner_score": 88.0,
            "best_low_quality_provider": None,
            "final_fetch_ok": True,
            "final_fetch_length": 900,
            "final_fetch_preview": "A",
            "final_fetch_metadata": {
                "fallback_mode": "reddit_json_summary",
                "json_recovery_provider": "tavily_json",
                "source_format": "reddit_json",
            },
            "providers": [
                {
                    "provider": "exa",
                    "status": "ok",
                    "elapsed_ms": 50.0,
                    "score": 88.0,
                    "is_low_quality": False,
                    "content_length": 900,
                    "heading_count": 4,
                    "substantive_line_count": 12,
                    "marker_hits": 6,
                    "domain_hits": 3,
                }
            ],
        }
    ]

    async def fake_run_benchmark(**_kwargs):
        return fake_results

    monkeypatch.setattr(benchmark.server, "_is_valid_web_url", lambda _url: True)
    monkeypatch.setattr(benchmark, "_run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(benchmark, "_apply_live_env_from_codex_config", lambda *_args, **_kwargs: (Path("/tmp/fake.toml"), ["EXA_API_KEY"]))

    json_out = tmp_path / "benchmark.json"
    args = benchmark.argparse.Namespace(
        url=["https://example.com/a"],
        url_file=None,
        providers="exa",
        preview_chars=180,
        concurrency=1,
        json_out=str(json_out),
        codex_config=None,
        no_codex_env=False,
        codex_env_override=False,
        artifact_dir=None,
    )

    rc = benchmark.asyncio.run(benchmark._async_main(args))

    assert rc == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["providers"] == ["exa"]
    assert payload["results"] == fake_results
    assert payload["summary"]["checked_url_count"] == 1
    assert payload["summary"]["winner_counts"] == {"exa": 1}
    assert payload["summary"]["provider_summary"]["exa"]["winner_count"] == 1
    assert payload["summary"]["final_fetch_fallback_mode_counts"] == {"reddit_json_summary": 1}
    assert payload["summary"]["final_fetch_recovery_provider_counts"] == {"tavily_json": 1}
    assert payload["summary"]["final_fetch_source_format_counts"] == {"reddit_json": 1}

    out = capsys.readouterr().out
    assert "Loaded Codex MCP env from /tmp/fake.toml: EXA_API_KEY" in out
    assert "final_fetch_fallback_modes: reddit_json_summary=1" in out
    assert "final_fetch_recovery_providers: tavily_json=1" in out
    assert "JSON written to:" in out
