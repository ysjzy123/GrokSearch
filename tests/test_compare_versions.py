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

    baseline_dir = compare_versions._archive_baseline(ROOT, "origin/grok-with-tavily")
    try:
        candidate = compare_versions._run_score_subprocess(ROOT, include_live=False)
        baseline = compare_versions._run_score_subprocess(baseline_dir, include_live=False)
        result = compare_versions.compare_scores(candidate, baseline)
    finally:
        shutil.rmtree(baseline_dir, ignore_errors=True)

    assert candidate["total_score"] > baseline["total_score"]
    assert candidate["categories"]["contract"]["score"] > baseline["categories"]["contract"]["score"]
    assert candidate["categories"]["cache_quality"]["score"] > baseline["categories"]["cache_quality"]["score"]
    assert candidate["categories"]["search_coverage"]["score"] > baseline["categories"]["search_coverage"]["score"]
    assert candidate["categories"]["aggregation_breadth"]["score"] > baseline["categories"]["aggregation_breadth"]["score"]
    assert candidate["categories"]["evidence_binding"]["score"] > baseline["categories"]["evidence_binding"]["score"]
    assert candidate["categories"]["citation_precision"]["score"] > baseline["categories"]["citation_precision"]["score"]
    assert candidate["categories"]["efficiency"]["score"] > baseline["categories"]["efficiency"]["score"]
    assert candidate["categories"]["source_quality"]["score"] > baseline["categories"]["source_quality"]["score"]
    assert candidate["categories"]["query_quality"]["score"] > baseline["categories"]["query_quality"]["score"]
    assert baseline["categories"]["engineering"]["score"] == 0
    assert result["accepted"] is True
