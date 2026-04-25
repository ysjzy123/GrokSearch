import asyncio
import importlib.util
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_versions.py"


def load_compare_versions():
    module_name = "compare_versions_test_module"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compare_scores_rejects_failed_gating_probes():
    compare_versions = load_compare_versions()

    candidate = {
        "total_score": 11,
        "gating_failures": ["validation:invalid_model_not_persisted"],
        "categories": {
            "validation": {"score": 6},
            "engineering": {"score": 5},
        },
    }
    baseline = {
        "total_score": 10,
        "gating_failures": [],
        "categories": {
            "validation": {"score": 6},
            "engineering": {"score": 4},
        },
    }

    result = compare_versions.compare_scores(candidate, baseline)

    assert result["accepted"] is False
    assert result["new_failed_gating_probes"] == ["validation:invalid_model_not_persisted"]
    assert result["remaining_failed_gating_probes"] == ["validation:invalid_model_not_persisted"]


def test_compare_scores_allows_historical_gating_failures_without_new_regressions():
    compare_versions = load_compare_versions()

    candidate = {
        "total_score": 12,
        "gating_failures": ["live:non_time_sources_available"],
        "categories": {
            "live": {"score": 14},
            "engineering": {"score": 6},
        },
    }
    baseline = {
        "total_score": 11,
        "gating_failures": ["live:non_time_sources_available"],
        "categories": {
            "live": {"score": 14},
            "engineering": {"score": 5},
        },
    }

    result = compare_versions.compare_scores(candidate, baseline)

    assert result["accepted"] is True
    assert result["new_failed_gating_probes"] == []
    assert result["remaining_failed_gating_probes"] == ["live:non_time_sources_available"]


def test_compare_versions_subprocess_isolates_baseline_scores():
    compare_versions = load_compare_versions()

    baseline_dir = Path(compare_versions.DEFAULT_BASELINE_PATH)
    try:
        candidate = compare_versions._run_score_subprocess(ROOT, include_live=False)
        baseline = compare_versions._run_score_subprocess(baseline_dir, include_live=False)
        result = compare_versions.compare_scores(candidate, baseline)
    finally:
        pass

    assert candidate["total_score"] > baseline["total_score"]
    assert candidate["categories"]["contract"]["score"] > baseline["categories"]["contract"]["score"]
    assert candidate["categories"]["cache_quality"]["score"] > baseline["categories"]["cache_quality"]["score"]
    assert candidate["categories"]["fetch_structuring"]["score"] > baseline["categories"].get("fetch_structuring", {}).get("score", 0)
    assert candidate["categories"]["search_coverage"]["score"] > baseline["categories"]["search_coverage"]["score"]
    assert candidate["categories"]["aggregation_breadth"]["score"] > baseline["categories"]["aggregation_breadth"]["score"]
    assert candidate["categories"]["evidence_binding"]["score"] > baseline["categories"]["evidence_binding"]["score"]
    assert candidate["categories"]["citation_precision"]["score"] > baseline["categories"]["citation_precision"]["score"]
    assert candidate["categories"]["efficiency"]["score"] > baseline["categories"]["efficiency"]["score"]
    assert candidate["categories"]["source_quality"]["score"] > baseline["categories"]["source_quality"]["score"]
    assert candidate["categories"]["query_quality"]["score"] > baseline["categories"]["query_quality"]["score"]
    assert baseline["categories"]["engineering"]["score"] == 0
    assert result["accepted"] is True


def test_load_live_env_from_codex_config_reads_grok_search_env(tmp_path):
    compare_versions = load_compare_versions()

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

    loaded = compare_versions._load_live_env_from_codex_config(config_path)

    assert loaded["GROK_API_URL"] == "https://example.test/v1"
    assert loaded["GROK_API_KEY"] == "grok-key"
    assert loaded["TAVILY_API_KEY"] == "t-key"
    assert loaded["FIRECRAWL_API_KEY"] == "f-key"
    assert loaded["EXA_API_KEY"] == "e-key"
    assert loaded["EXA_ENABLED"] == "true"


def test_prepare_live_env_strips_proxy_and_injects_codex_values(monkeypatch, tmp_path):
    compare_versions = load_compare_versions()

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[mcp_servers.grok-search.env]
GROK_API_URL = "https://example.test/v1"
GROK_API_KEY = "grok-key"
TAVILY_API_KEY = "t-key"
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    monkeypatch.delenv("GROK_API_URL", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    prepared = compare_versions._prepare_live_env(prefer_existing=True, config_path=config_path)

    assert "HTTP_PROXY" not in prepared
    assert "HTTPS_PROXY" not in prepared
    assert prepared["GROK_API_URL"] == "https://example.test/v1"
    assert prepared["GROK_API_KEY"] == "grok-key"
    assert prepared["TAVILY_API_KEY"] == "t-key"


def test_extract_final_fetch_metadata_parses_structured_fallback_fields():
    compare_versions = load_compare_versions()

    text = """
---
source_url: https://www.reddit.com/r/Python/comments/1spdhgt/sunday_daily_thread_whats_everyone_working_on/
inferred_title: Sunday Daily Thread: What's everyone working on this week?
fetch_structure_version: 1
---

# Sunday Daily Thread: What's everyone working on this week?

- fallback_mode: reddit_json_summary
- source_format: reddit_json
- json_recovery_provider: tavily_json
""".strip()

    metadata = compare_versions._extract_final_fetch_metadata(text)

    assert metadata == {
        "source_url": "https://www.reddit.com/r/Python/comments/1spdhgt/sunday_daily_thread_whats_everyone_working_on/",
        "inferred_title": "Sunday Daily Thread: What's everyone working on this week?",
        "fetch_structure_version": 1,
        "fallback_mode": "reddit_json_summary",
        "source_format": "reddit_json",
        "json_recovery_provider": "tavily_json",
    }


def test_compare_live_fetch_benchmark_includes_structured_fallback_deltas():
    compare_versions = load_compare_versions()

    candidate = {
        "live_fetch_benchmark": {
            "checked_urls": ["https://example.com/a"],
            "success_rate": 1.0,
            "final_fetch_success_rate": 1.0,
            "final_fetch_structured_rate": 1.0,
            "final_fetch_fallback_rate": 1.0,
            "all_low_quality_rate": 0.0,
            "winner_counts": {"exa": 1},
            "final_fetch_fallback_mode_counts": {"reddit_json_summary": 1},
            "final_fetch_source_format_counts": {"reddit_json": 1},
            "final_fetch_recovery_provider_counts": {"tavily_json": 1},
            "provider_summary": {
                "exa": {
                    "success_rate": 1.0,
                    "non_low_quality_rate": 1.0,
                    "winner_count": 1,
                    "avg_content_length": 1200.0,
                    "avg_elapsed_s": 0.8,
                }
            },
            "has_selector": True,
        }
    }
    baseline = {
        "live_fetch_benchmark": {
            "checked_urls": ["https://example.com/a"],
            "success_rate": 1.0,
            "final_fetch_success_rate": 1.0,
            "final_fetch_structured_rate": 0.0,
            "final_fetch_fallback_rate": 0.0,
            "all_low_quality_rate": 0.0,
            "winner_counts": {"exa": 1},
            "final_fetch_fallback_mode_counts": {},
            "final_fetch_source_format_counts": {},
            "final_fetch_recovery_provider_counts": {},
            "provider_summary": {
                "exa": {
                    "success_rate": 1.0,
                    "non_low_quality_rate": 1.0,
                    "winner_count": 1,
                    "avg_content_length": 1000.0,
                    "avg_elapsed_s": 0.9,
                }
            },
            "has_selector": True,
        }
    }

    result = compare_versions.compare_live_fetch_benchmark(candidate, baseline)

    assert result is not None
    assert result["candidate_final_fetch_structured_rate"] == 1.0
    assert result["baseline_final_fetch_structured_rate"] == 0.0
    assert result["final_fetch_structured_rate_delta"] == 1.0
    assert result["candidate_final_fetch_fallback_mode_counts"] == {"reddit_json_summary": 1}
    assert result["candidate_final_fetch_source_format_counts"] == {"reddit_json": 1}
    assert result["candidate_final_fetch_recovery_provider_counts"] == {"tavily_json": 1}


def test_collect_live_fetch_benchmark_metrics_collects_final_fetch_metadata(tmp_path):
    compare_versions = load_compare_versions()

    benchmark_file = tmp_path / "urls.txt"
    benchmark_file.write_text("https://example.com/reddit\n", encoding="utf-8")

    class FakeConfig:
        tavily_enabled = True
        tavily_api_key = "t-key"
        firecrawl_api_key = "f-key"
        exa_enabled = True
        exa_api_key = "e-key"

    class FakeServer:
        config = FakeConfig()

        @staticmethod
        async def _call_tavily_extract(_url):
            return "# Tavily\n\nShort shell."

        @staticmethod
        async def _call_firecrawl_scrape(_url):
            return "# Firecrawl\n\nMore content here."

        @staticmethod
        async def _call_exa_contents(_url):
            return "# Exa\n\nBest content here."

        @staticmethod
        def _build_fetch_candidate(provider, text, url):
            return {
                "provider": provider,
                "analysis": {"content_length": len(text)},
                "is_low_quality": provider == "tavily",
                "score": {"tavily": 10.0, "firecrawl": 20.0, "exa": 30.0}[provider],
            }

        @staticmethod
        def _select_best_fetch_candidate(candidates):
            return max(candidates, key=lambda item: item["score"]) if candidates else None

        @staticmethod
        async def web_fetch(_url):
            return "\n".join(
                [
                    "---",
                    "source_url: https://example.com/reddit",
                    "inferred_title: Example Reddit Thread",
                    "fetch_structure_version: 1",
                    "---",
                    "",
                    "# Example Reddit Thread",
                    "",
                    "- fallback_mode: reddit_json_summary",
                    "- source_format: reddit_json",
                    "- json_recovery_provider: tavily_json",
                ]
            )

    metrics = compare_versions._collect_live_fetch_benchmark_metrics
    loaded = asyncio.run(metrics(FakeServer(), benchmark_file))

    assert loaded is not None
    assert loaded["final_fetch_structured_urls"] == 1
    assert loaded["final_fetch_structured_rate"] == 1.0
    assert loaded["final_fetch_fallback_urls"] == 1
    assert loaded["final_fetch_fallback_mode_counts"] == {"reddit_json_summary": 1}
    assert loaded["final_fetch_source_format_counts"] == {"reddit_json": 1}
    assert loaded["final_fetch_recovery_provider_counts"] == {"tavily_json": 1}
    assert loaded["summaries"][0]["final_fetch_metadata"] == {
        "source_url": "https://example.com/reddit",
        "inferred_title": "Example Reddit Thread",
        "fetch_structure_version": 1,
        "fallback_mode": "reddit_json_summary",
        "source_format": "reddit_json",
        "json_recovery_provider": "tavily_json",
    }
