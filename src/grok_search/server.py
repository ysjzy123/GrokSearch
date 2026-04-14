import sys
from pathlib import Path
import re
from collections import Counter

# 支持直接运行：添加 src 目录到 Python 路径
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fastmcp import FastMCP, Context
from typing import Annotated, Any, Optional
from pydantic import Field, ValidationError
from urllib.parse import urlparse

# 尝试使用绝对导入（支持 mcp run）
try:
    from grok_search.providers.grok import GrokSearchProvider
    from grok_search.logger import log_info
    from grok_search.config import config
    from grok_search.sources import SourcesCache, merge_sources, new_session_id, split_answer_and_sources
    from grok_search.planning import (
        ComplexityOutput,
        ExecutionOrderOutput,
        IntentOutput,
        SearchTerm,
        StrategyOutput,
        SubQuery,
        ToolPlanItem,
        engine as planning_engine,
        _split_csv,
    )
except ImportError:
    from .providers.grok import GrokSearchProvider
    from .logger import log_info
    from .config import config
    from .sources import SourcesCache, merge_sources, new_session_id, split_answer_and_sources
    from .planning import (
        ComplexityOutput,
        ExecutionOrderOutput,
        IntentOutput,
        SearchTerm,
        StrategyOutput,
        SubQuery,
        ToolPlanItem,
        engine as planning_engine,
        _split_csv,
    )

import asyncio

mcp = FastMCP("grok-search")

_SOURCES_CACHE = SourcesCache(max_size=256, persist_dir=config.sources_cache_dir)
_AVAILABLE_MODELS_CACHE: dict[tuple[str, str], list[str]] = {}
_AVAILABLE_MODELS_LOCK = asyncio.Lock()
_ASCII_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+#.\-]{1,}")
_CJK_SEGMENT_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
_CLAIM_SPLIT_PATTERN = re.compile(r"(?<=[。！？.!?;；])\s+")
_INLINE_CITATION_URL_PATTERN = re.compile(r"\[\[\d+\]\]\((https?://[^)]+)\)")
_MAX_PLANNED_QUERIES = 3
_INITIAL_EXTRA_SOURCE_BUDGET_CAP = 6
_MIN_SOURCES_FOR_EARLY_STOP = 4
_MIN_DOMAIN_COUNT_FOR_EARLY_STOP = 2
_MIN_SOURCES_FOR_EARLY_STOP_COMPLEX = 6
_MIN_DOMAIN_COUNT_FOR_EARLY_STOP_COMPLEX = 4
_MIN_SUPPORTED_CLAIMS_FOR_EARLY_STOP_COMPLEX = 3
_MAX_ENRICHABLE_SOURCES = 2
_MAX_RANKABLE_SOURCES = 8
_FAST_MODEL_SUFFIX = "-fast"
_AUTO_MODEL_SUFFIX = "-auto"
_MIN_SOURCES_FOR_RANKING = 6
_MAX_FOLLOWUP_QUERIES_FOR_MULTIFACET = 2
_MAX_EXPANSION_EXTRA_SOURCE_BUDGET = 6
_EXTERNAL_SEARCH_TIMEOUT_SECONDS = 8.0
_EXPANSION_EXTERNAL_SEARCH_TIMEOUT_SECONDS = 6.0
_RELAXED_EXTERNAL_SEARCH_TIMEOUT_SECONDS = 4.0
_SOURCE_ENRICH_TIMEOUT_SECONDS = 8.0
_SOURCE_RANK_TIMEOUT_SECONDS = 4.0
_SOURCE_SYNTHESIS_TIMEOUT_SECONDS = 18.0
_MAX_SYNTHESIS_SOURCES = 8
_AUTHORITATIVE_SOURCE_DOMAINS = {
    "fastapi.tiangolo.com",
    "react.dev",
    "vuejs.org",
    "roadmap.sh",
    "nextjs.org",
    "nuxt.com",
    "vite.dev",
    "vitejs.dev",
    "docs.python.org",
    "wikipedia.org",
    "fastapi-tutorial.readthedocs.io",
}
_ENGLISH_COMPARISON_SEPARATORS = (" vs ", " versus ")
_LEARNING_QUERY_KEYWORDS = ("learn", "learning", "guide", "tutorial", "roadmap")
_TIME_SENSITIVE_QUERY_KEYWORDS = ("latest", "current", "stable", "version", "versions", "release")
_SUBJECT_EXTRACTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "architecture",
    "best",
    "between",
    "choose",
    "compare",
    "comparison",
    "current",
    "docs",
    "for",
    "guide",
    "how",
    "i",
    "in",
    "learn",
    "learning",
    "latest",
    "official",
    "or",
    "pick",
    "practices",
    "release",
    "roadmap",
    "should",
    "stable",
    "the",
    "to",
    "tutorial",
    "version",
    "versions",
    "versus",
    "vs",
    "what",
    "which",
    "with",
}
_SUBJECT_QUERY_HINTS: dict[str, dict[str, list[str] | str]] = {
    "react": {
        "label": "React 19",
        "domain": "react.dev",
        "extras": ["learn", "official guide", "Next.js", "TypeScript", "Vite"],
    },
    "vue": {
        "label": "Vue 3",
        "domain": "vuejs.org",
        "extras": ["guide", "official docs", "Nuxt", "TypeScript", "Vite"],
    },
    "next.js": {
        "label": "Next.js",
        "domain": "nextjs.org",
        "extras": ["App Router", "official docs", "TypeScript", "release notes"],
    },
    "nextjs": {
        "label": "Next.js",
        "domain": "nextjs.org",
        "extras": ["App Router", "official docs", "TypeScript", "release notes"],
    },
    "nuxt": {
        "label": "Nuxt",
        "domain": "nuxt.com",
        "extras": ["Vue 3", "official docs", "TypeScript", "release notes"],
    },
}


def _source_authority_score(source: dict) -> float:
    url = (source.get("url") or "").strip()
    domain = urlparse(url).netloc.lower()
    if not domain:
        return 0.0
    if any(domain == item or domain.endswith(f".{item}") for item in _AUTHORITATIVE_SOURCE_DOMAINS):
        return 2.0
    if domain.endswith('.readthedocs.io') or domain.startswith('docs.'):
        return 1.2
    if domain.endswith('.org'):
        return 0.6
    return 0.0


def _source_text_blob(source: dict) -> str:
    return " ".join(
        str(source.get(key) or "")
        for key in ("title", "description", "url", "query_used")
    ).lower()


def _source_contains_keywords(source: dict, keywords: list[str]) -> bool:
    haystack = _source_text_blob(source)
    return any(keyword.lower() in haystack for keyword in keywords)


def _sources_from_inline_citations(answer: str) -> list[dict]:
    seen: set[str] = set()
    sources: list[dict] = []
    for url in _INLINE_CITATION_URL_PATTERN.findall(answer or ""):
        normalized = url.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        sources.append({
            "url": normalized,
            "title": normalized,
            "provider": "grok",
        })
    return sources


async def _fetch_available_models(api_url: str, api_key: str) -> list[str]:
    import httpx

    models_url = f"{api_url.rstrip('/')}/models"
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            models_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        data = response.json()

    models: list[str] = []
    for item in (data or {}).get("data", []) or []:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            models.append(item["id"])
    return models


async def _get_available_models_cached(api_url: str, api_key: str) -> list[str]:
    key = (api_url, api_key)
    async with _AVAILABLE_MODELS_LOCK:
        if key in _AVAILABLE_MODELS_CACHE:
            return _AVAILABLE_MODELS_CACHE[key]

    try:
        models = await _fetch_available_models(api_url, api_key)
    except Exception:
        models = []

    async with _AVAILABLE_MODELS_LOCK:
        _AVAILABLE_MODELS_CACHE[key] = models
    return models


def _validation_error(message: str, details: list[dict] | None = None) -> str:
    import json

    payload: dict = {"error": "validation_error", "message": message}
    if details:
        payload["details"] = details
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _word_count(text: str) -> int:
    return len([part for part in text.strip().split() if part])


def _is_valid_web_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _find_project_root(start: Path) -> Path:
    current = start.resolve()
    while True:
        if (current / ".git").exists() or (current / ".claude").exists():
            return current
        if current.parent == current:
            return start.resolve()
        current = current.parent


def _split_extra_sources_budget(
    extra_sources: int,
    has_tavily: bool,
    has_firecrawl: bool,
) -> tuple[int, int]:
    if extra_sources <= 0:
        return 0, 0
    if has_firecrawl and has_tavily:
        firecrawl_count = extra_sources // 2
        tavily_count = extra_sources - firecrawl_count
        return tavily_count, firecrawl_count
    if has_tavily:
        return extra_sources, 0
    if has_firecrawl:
        return 0, extra_sources
    return 0, 0


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_SEGMENT_PATTERN.search(text))


def _dedupe_queries(queries: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = " ".join((query or "").split()).strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _should_expand_search_query(query: str, extra_sources: int) -> bool:
    if extra_sources < 6:
        return False

    query_lower = query.lower()
    keywords = [
        " vs ",
        "versus",
        "compare",
        "comparison",
        "tradeoff",
        "guide",
        "tutorial",
        "roadmap",
        "best practices",
        "architecture",
        "alternatives",
        "recommend",
        "学习",
        "教程",
        "入门",
        "路线",
        "指南",
        "最佳实践",
        "对比",
        "区别",
        "方案",
        "架构",
        "推荐",
        "替代",
    ]
    return any(keyword in query_lower or keyword in query for keyword in keywords)


def _is_multifacet_search_query(query: str) -> bool:
    query_lower = query.lower()
    keywords = [
        " vs ",
        "versus",
        "compare",
        "comparison",
        "tradeoff",
        "tradeoffs",
        "learn",
        "learning",
        "guide",
        "tutorial",
        "roadmap",
        "best practices",
        "alternatives",
        "学习",
        "教程",
        "入门",
        "路线",
        "指南",
        "最佳实践",
        "对比",
        "区别",
        "优缺点",
        "替代",
    ]
    return any(keyword in query_lower or keyword in query for keyword in keywords)


def _is_learning_search_query(query: str) -> bool:
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in _LEARNING_QUERY_KEYWORDS) or any(
        keyword in query for keyword in ["学习", "教程", "入门", "路线", "指南"]
    )


def _is_time_sensitive_search_query(query: str) -> bool:
    query_lower = query.lower()
    if re.search(r"\b20\d{2}\b", query_lower):
        return True
    return any(keyword in query_lower for keyword in _TIME_SENSITIVE_QUERY_KEYWORDS) or any(
        keyword in query for keyword in ["最新", "版本", "今年", "现在", "发布"]
    )


def _preferred_analysis_model(base_model: str) -> str:
    normalized = (base_model or "").strip()
    if not normalized:
        return normalized
    suffix = ""
    if normalized.endswith(":online"):
        normalized, suffix = normalized[:-7], ":online"
    if normalized.endswith(_AUTO_MODEL_SUFFIX):
        return normalized + suffix
    if normalized.endswith(_FAST_MODEL_SUFFIX):
        return normalized[: -len(_FAST_MODEL_SUFFIX)] + _AUTO_MODEL_SUFFIX + suffix
    return normalized + suffix


def _should_use_analysis_model(
    query: str,
    *,
    prefer_source_synthesis: bool,
    planned_queries: list[str],
) -> bool:
    if prefer_source_synthesis:
        return True
    if len(planned_queries) > 1:
        return True
    return _is_multifacet_search_query(query) or _is_learning_search_query(query)


async def _resolve_stage_models(
    api_url: str,
    api_key: str,
    base_model: str,
    *,
    query: str,
    prefer_source_synthesis: bool,
    planned_queries: list[str],
) -> dict[str, str]:
    models = {"search": base_model, "analysis": base_model}
    candidate_analysis_model = _preferred_analysis_model(base_model)
    if candidate_analysis_model == base_model:
        return models
    if not _should_use_analysis_model(
        query,
        prefer_source_synthesis=prefer_source_synthesis,
        planned_queries=planned_queries,
    ):
        return models
    available = await _get_available_models_cached(api_url, api_key)
    if not available or candidate_analysis_model in available:
        models["analysis"] = candidate_analysis_model
    return models


def _extract_ascii_subject_segment(text: str, *, from_end: bool) -> str:
    tokens = _ASCII_TOKEN_PATTERN.findall(text)
    if not tokens:
        return ""

    ordered = list(reversed(tokens)) if from_end else tokens
    selected: list[str] = []
    for token in ordered:
        token_lower = token.casefold()
        if re.fullmatch(r"20\d{2}", token):
            if selected:
                break
            continue
        if token_lower in _SUBJECT_EXTRACTION_STOPWORDS:
            if selected:
                break
            continue
        selected.append(token)
        if len(selected) >= 3:
            break

    if not selected:
        return ""
    if from_end:
        selected.reverse()
    return " ".join(selected)


def _extract_comparison_subjects(query: str) -> tuple[str, str] | None:
    normalized = " ".join((query or "").split())
    query_lower = normalized.lower()
    for separator in _ENGLISH_COMPARISON_SEPARATORS:
        index = query_lower.find(separator)
        if index == -1:
            continue
        left = _extract_ascii_subject_segment(normalized[:index], from_end=True)
        right = _extract_ascii_subject_segment(
            normalized[index + len(separator):],
            from_end=False,
        )
        if left and right and left.casefold() != right.casefold():
            return left, right
    return None


def _subject_focus_terms(hint: dict[str, list[str] | str], *, learning: bool) -> tuple[str, str]:
    extras = [str(item).strip() for item in hint.get("extras", []) if str(item).strip()]
    guide_term = ""
    ecosystem_term = ""
    for item in extras:
        lowered = item.casefold()
        if not guide_term and any(keyword in lowered for keyword in ["learn", "guide", "docs", "tutorial"]):
            guide_term = item
            continue
        if lowered in {"typescript", "vite", "official docs", "official guide", "release notes"}:
            continue
        if not ecosystem_term:
            ecosystem_term = item
    if not learning:
        guide_term = ""
    return guide_term, ecosystem_term


def _build_subject_focus_query(subject: str, *, learning: bool, time_sensitive: bool) -> str:
    normalized = subject.strip()
    hint = _SUBJECT_QUERY_HINTS.get(normalized.casefold())

    if hint:
        guide_term, ecosystem_term = _subject_focus_terms(hint, learning=learning)
        parts = [str(hint["label"]), str(hint["domain"]), "official docs"]
        if guide_term:
            parts.append(guide_term)
        if time_sensitive:
            parts.append("current version")
        if ecosystem_term:
            parts.append(ecosystem_term)
        return " ".join(parts)

    parts = [normalized, "official docs"]
    if learning:
        parts.append("guide")
    if time_sensitive:
        parts.append("current version")
    return " ".join(parts)


def _build_relaxed_search_query(query: str) -> str:
    normalized = " ".join((query or "").split()).strip()
    if not normalized:
        return ""

    lowered = normalized.casefold()
    for key, hint in _SUBJECT_QUERY_HINTS.items():
        label = str(hint.get("label") or "").strip()
        domain = str(hint.get("domain") or "").strip()
        candidates = [key.casefold(), label.casefold(), domain.casefold()]
        if any(candidate and candidate in lowered for candidate in candidates):
            return " ".join(part for part in [label or key, domain, "official docs"] if part)

    if _contains_cjk(normalized):
        if "官方文档" in normalized:
            return normalized.split("官方文档", 1)[0].strip() + " 官方文档"
        return normalized

    noise = {
        "learning",
        "roadmap",
        "guide",
        "tutorial",
        "current",
        "version",
        "versions",
        "official",
        "docs",
        "learn",
        "typescript",
        "vite",
        "benchmarks",
        "ecosystem",
        "release",
        "notes",
        "stable",
    }
    filtered = [token for token in normalized.split() if token.casefold() not in noise]
    if not filtered:
        return normalized
    return " ".join(filtered[:5] + (["official", "docs"] if "official docs" in lowered else []))


def _build_search_queries(query: str, extra_sources: int) -> list[str]:
    base = query.strip()
    if not _should_expand_search_query(base, extra_sources):
        return [base]

    query_lower = base.lower()
    is_learning_query = _is_learning_search_query(base)
    is_time_sensitive_query = _is_time_sensitive_search_query(base)
    comparison_subjects = _extract_comparison_subjects(base)
    if comparison_subjects and is_learning_query:
        left_subject, right_subject = comparison_subjects
        return _dedupe_queries(
            [
                base,
                _build_subject_focus_query(
                    left_subject,
                    learning=True,
                    time_sensitive=is_time_sensitive_query,
                ),
                _build_subject_focus_query(
                    right_subject,
                    learning=True,
                    time_sensitive=is_time_sensitive_query,
                ),
            ]
        )[:_MAX_PLANNED_QUERIES]

    if _contains_cjk(base):
        if any(keyword in query_lower or keyword in base for keyword in ["对比", "区别", "选哪个"]):
            return _dedupe_queries([base, f"{base} 官方文档", f"{base} 性能对比"])
        if any(keyword in query_lower or keyword in base for keyword in ["学习", "入门", "教程", "路线", "指南"]):
            return _dedupe_queries([base, f"{base} 官方文档", f"{base} 学习路线"])
        return _dedupe_queries([base, f"{base} 官方文档", f"{base} 最新版本"])

    if any(keyword in query_lower for keyword in [" vs ", "versus", "compare", "comparison"]):
        return _dedupe_queries([base, f"{base} official docs", f"{base} benchmarks"])[:_MAX_PLANNED_QUERIES]
    if any(keyword in query_lower for keyword in ["learn", "learning", "guide", "tutorial", "roadmap"]):
        return _dedupe_queries([base, f"{base} official docs", f"{base} roadmap current versions"])[:_MAX_PLANNED_QUERIES]
    return _dedupe_queries([base, f"{base} official docs", f"{base} current version ecosystem"])[:_MAX_PLANNED_QUERIES]


def _local_source_priority_score(query: str, source: dict) -> float:
    url = (source.get("url") or "").strip().lower()
    domain = urlparse(url).netloc.lower()
    query_used = " ".join((source.get("query_used") or "").split()).strip().casefold()
    base_query = " ".join((query or "").split()).strip().casefold()

    score = _source_authority_score(source) * 3.0
    if query_used and query_used != base_query:
        score += 0.8

    if _is_multifacet_search_query(query) and _source_contains_keywords(source, ["roadmap", "guide", "official docs"]):
        score += 0.9

    if _is_time_sensitive_search_query(query) and _source_contains_keywords(
        source,
        ["release", "current version", "stable", "2026", "react 19", "vue 3", "next.js", "nuxt", "vite", "typescript"],
    ):
        score += 0.75

    if domain == "roadmap.sh":
        score += 1.4
    if any(domain == item or domain.endswith(f".{item}") for item in ["react.dev", "vuejs.org", "nextjs.org", "nuxt.com", "vite.dev", "vitejs.dev"]):
        score += 1.8

    if domain == "medium.com" or domain.endswith('.medium.com'):
        score -= 0.2
    elif domain.startswith('blog.') or '/blog/' in url:
        score -= 0.1

    source_score = source.get("score")
    if isinstance(source_score, (int, float)):
        score += min(float(source_score), 1.0) * 0.3

    return score


def _should_prioritize_sources_locally(query: str, sources: list[dict]) -> bool:
    return len(sources) > 1 and (_is_multifacet_search_query(query) or _is_time_sensitive_search_query(query))


def _prioritize_sources_locally(query: str, sources: list[dict]) -> tuple[list[dict], bool]:
    if not _should_prioritize_sources_locally(query, sources):
        return sources, False

    base_query = " ".join((query or "").split()).strip().casefold()
    ordered = sorted(
        sources,
        key=lambda source: (
            _local_source_priority_score(query, source),
            (" ".join((source.get("query_used") or "").split()).strip().casefold() != base_query),
            len(source.get("description") or ""),
        ),
        reverse=True,
    )
    changed = [item.get("url") for item in ordered] != [item.get("url") for item in sources]
    return ordered, changed


def _select_initial_extra_source_budget(extra_sources: int, planned_queries: list[str]) -> int:
    if extra_sources <= 0:
        return 0
    return min(extra_sources, _INITIAL_EXTRA_SOURCE_BUDGET_CAP)


def _select_expansion_extra_source_budget(query: str, remaining_budget: int, followup_queries: list[str]) -> int:
    if remaining_budget <= 0 or not followup_queries:
        return 0
    return min(remaining_budget, _MAX_EXPANSION_EXTRA_SOURCE_BUDGET)


def _allocate_query_budgets(total: int, query_count: int) -> list[int]:
    if total <= 0 or query_count <= 0:
        return [0] * max(query_count, 0)

    active_queries = min(query_count, total)
    budgets = [0] * query_count
    base_share, remainder = divmod(total, active_queries)
    for index in range(active_queries):
        budgets[index] = base_share + (1 if index < remainder else 0)
    return budgets


def _source_domains(sources: list[dict]) -> list[str]:
    domains: set[str] = set()
    for source in sources:
        url = (source.get("url") or "").strip()
        if not url:
            continue
        netloc = urlparse(url).netloc.lower()
        if netloc:
            domains.add(netloc)
    return sorted(domains)


def _build_search_trace(
    root_query: str,
    planned_queries: list[str],
    executed_queries: list[dict],
    sources: list[dict],
    *,
    requested_budget: int = 0,
    used_budget: int = 0,
    phases: list[dict] | None = None,
    decision: dict[str, Any] | None = None,
    postprocessing: dict[str, Any] | None = None,
) -> dict:
    provider_counts = Counter(
        record.get("provider")
        for record in executed_queries
        if isinstance(record, dict) and record.get("provider")
    )
    executed_query_values = {
        record.get("query", "").strip()
        for record in executed_queries
        if isinstance(record, dict) and record.get("query")
    }
    followup_executed = any(
        isinstance(record, dict) and record.get("phase") == "expansion"
        for record in executed_queries
    )
    return {
        "root_query": root_query,
        "planned_queries": planned_queries,
        "executed_queries": executed_queries,
        "phases": phases or [],
        "decision": decision or {},
        "postprocessing": postprocessing or {},
        "summary": {
            "planned_query_count": len(planned_queries),
            "executed_query_count": len(executed_query_values),
            "provider_counts": dict(provider_counts),
            "sources_count": len(sources),
            "domain_count": len(_source_domains(sources)),
            "expanded": followup_executed,
            "planned_expansion": len(planned_queries) > 1,
            "external_task_count": sum(
                1
                for record in executed_queries
                if isinstance(record, dict) and record.get("provider") in {"tavily", "firecrawl"}
            ),
            "budget_requested": max(requested_budget, 0),
            "budget_used": max(min(used_budget, requested_budget), 0),
            "budget_unused": max(requested_budget - used_budget, 0),
            "early_stopped": bool(decision and decision.get("early_stopped")),
            "stop_reason": (decision or {}).get("reason", ""),
            "followup_executed": followup_executed,
        },
    }


def _normalize_cached_search_payload(payload: Any) -> dict:
    if isinstance(payload, list):
        return {"sources": [item for item in payload if isinstance(item, dict)]}

    if not isinstance(payload, dict):
        return {"sources": []}

    normalized: dict[str, Any] = {
        "sources": [item for item in payload.get("sources", []) if isinstance(item, dict)]
    }

    evidence_bindings = payload.get("evidence_bindings")
    if isinstance(evidence_bindings, list):
        normalized["evidence_bindings"] = [
            item for item in evidence_bindings if isinstance(item, dict)
        ]

    search_trace = payload.get("search_trace")
    if isinstance(search_trace, dict):
        normalized["search_trace"] = search_trace

    return normalized


def _normalize_claim_text(text: str) -> str:
    cleaned = re.sub(r"^\s*(?:[-*]|\d+\.)\s*", "", (text or "").strip())
    cleaned = re.sub(r"^\s*#+\s*", "", cleaned)
    cleaned = re.sub(r"\[\[\d+\]\]\(https?://[^)]+\)", "", cleaned)
    cleaned = re.sub(r"[*_`]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _extract_claims(answer: str, max_claims: int = 12) -> list[str]:
    lines = [_normalize_claim_text(line) for line in (answer or "").splitlines()]
    claims = [line for line in lines if len(line) >= 12]

    if len(claims) <= 1:
        candidates = [
            _normalize_claim_text(part)
            for part in _CLAIM_SPLIT_PATTERN.split(answer or "")
        ]
        claims = [part for part in candidates if len(part) >= 12]

    deduped: list[str] = []
    seen: set[str] = set()
    for claim in claims:
        key = claim.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(claim)
        if len(deduped) >= max_claims:
            break
    return deduped


def _match_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    lowered = (text or "").lower()

    for match in _ASCII_TOKEN_PATTERN.finditer(lowered):
        tokens.add(match.group(0))

    for match in _CJK_SEGMENT_PATTERN.finditer(text or ""):
        segment = match.group(0)
        if len(segment) <= 2:
            tokens.add(segment)
            continue
        tokens.add(segment)
        for index in range(len(segment) - 1):
            tokens.add(segment[index : index + 2])

    return tokens


def _ordered_match_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    lowered = (text or "").lower()

    for match in _ASCII_TOKEN_PATTERN.finditer(lowered):
        tokens.append(match.group(0))

    for match in _CJK_SEGMENT_PATTERN.finditer(text or ""):
        segment = match.group(0)
        if len(segment) <= 2:
            tokens.append(segment)
            continue
        tokens.append(segment)

    return tokens


def _claim_phrases(text: str) -> set[str]:
    ordered = _ordered_match_tokens(text)
    phrases: set[str] = set()
    for left, right in zip(ordered, ordered[1:]):
        if left == right:
            continue
        phrases.add(f"{left} {right}")
    return phrases


def _score_overlap_tokens(claim_tokens: set[str], source_token_sets: list[set[str]]) -> dict[str, float]:
    frequencies: Counter[str] = Counter()
    for token in claim_tokens:
        frequencies[token] = sum(1 for token_set in source_token_sets if token in token_set)

    weights: dict[str, float] = {}
    for token in claim_tokens:
        frequency = max(frequencies[token], 1)
        weight = 1.0 / frequency
        if len(token) >= 5:
            weight += 0.25
        elif _contains_cjk(token):
            weight += 0.2
        weights[token] = weight
    return weights


def _build_evidence_bindings(answer: str, sources: list[dict]) -> list[dict]:
    claims = _extract_claims(answer)
    if not claims or not sources:
        return []

    bindings: list[dict] = []
    source_tokens: list[tuple[dict, set[str], set[str], str]] = []
    for source in sources:
        title = source.get("title") or source.get("url") or ""
        description = source.get("description") or ""
        title_tokens = _match_tokens(title)
        combined_tokens = title_tokens | _match_tokens(description)
        normalized_text = " ".join(f"{title} {description}".lower().split())
        source_tokens.append((source, title_tokens, combined_tokens, normalized_text))

    for claim in claims:
        claim_tokens = _match_tokens(claim)
        if not claim_tokens:
            continue

        claim_phrases = _claim_phrases(claim)
        token_weights = _score_overlap_tokens(
            claim_tokens,
            [combined_tokens for _source, _title_tokens, combined_tokens, _text in source_tokens],
        )
        ranked: list[tuple[float, dict, list[str]]] = []
        for source, title_tokens, combined_tokens, normalized_text in source_tokens:
            overlap = claim_tokens & combined_tokens
            if not overlap:
                continue
            title_overlap = claim_tokens & title_tokens
            weighted_overlap = sum(token_weights.get(token, 1.0) for token in overlap)
            title_bonus = sum(token_weights.get(token, 1.0) for token in title_overlap) * 0.5
            phrase_bonus = sum(0.75 for phrase in claim_phrases if phrase in normalized_text)
            exact_claim_bonus = 1.0 if " ".join(claim.lower().split()) in normalized_text else 0.0
            authority_bonus = _source_authority_score(source)
            score = float(weighted_overlap + title_bonus + phrase_bonus + exact_claim_bonus + authority_bonus)
            ranked.append((score, source, sorted(overlap)[:6]))

        ranked.sort(
            key=lambda item: (
                item[0],
                len(item[1].get("description") or ""),
                item[1].get("provider") == "tavily",
            ),
            reverse=True,
        )

        if not ranked:
            continue

        top_score = ranked[0][0]
        support = []
        for score, source, matched_terms in ranked[:2]:
            if score < max(1.75, top_score * 0.85):
                continue
            support.append(
                {
                    "url": source.get("url"),
                    "title": source.get("title") or source.get("url"),
                    "provider": source.get("provider"),
                    "matched_terms": matched_terms,
                    "score": round(score, 2),
                }
            )

        bindings.append({"claim": claim, "sources": support})

    return bindings


def _format_binding_citations(binding_sources: list[dict], sources: list[dict]) -> str:
    url_to_index = {
        (source.get("url") or "").strip(): idx
        for idx, source in enumerate(sources, 1)
        if (source.get("url") or "").strip()
    }
    ranked_sources = sorted(
        binding_sources,
        key=lambda item: (
            _source_authority_score(item),
            float(item.get("score") or 0.0),
            item.get("provider") == "tavily",
        ),
        reverse=True,
    )
    parts: list[str] = []
    seen: set[str] = set()
    for item in ranked_sources[:2]:
        url = (item.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        index = url_to_index.get(url)
        if index is not None:
            parts.append(f"[[{index}]]({url})")
    return " ".join(parts)


def _attach_evidence_citations(answer: str, sources: list[dict]) -> tuple[str, list[dict]]:
    bindings = _build_evidence_bindings(answer, sources)
    if not answer or not bindings:
        return answer, bindings

    claim_to_citations: dict[str, str] = {}
    for binding in bindings:
        claim = _normalize_claim_text(binding.get("claim") or "")
        citations = _format_binding_citations(binding.get("sources") or [], sources)
        if claim and citations:
            claim_to_citations[claim.casefold()] = citations

    if not claim_to_citations:
        return answer, bindings

    rendered: list[str] = []
    in_code_block = False
    updated = False
    for line in answer.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            rendered.append(line)
            continue
        if in_code_block or not stripped or "[[" in line:
            rendered.append(line)
            continue

        normalized = _normalize_claim_text(line)
        citations = claim_to_citations.get(normalized.casefold())
        if citations:
            rendered.append(f"{line} {citations}")
            updated = True
        else:
            rendered.append(line)

    if updated:
        return "\n".join(rendered), _build_evidence_bindings("\n".join(rendered), sources)
    return answer, bindings


def _summarize_source_support(answer: str, sources: list[dict]) -> dict[str, int]:
    claims = _extract_claims(answer)
    bindings = _build_evidence_bindings(answer, sources)
    supported_claims = sum(1 for item in bindings if item.get("sources"))
    return {
        "sources_count": len(sources),
        "domain_count": len(_source_domains(sources)),
        "claim_count": len(claims),
        "supported_claims": supported_claims,
    }


def _should_expand_after_initial(
    query: str,
    answer: str,
    sources: list[dict],
    followup_queries: list[str],
    remaining_budget: int,
    grok_failed: bool,
) -> tuple[bool, dict[str, int], str]:
    support = _summarize_source_support(answer, sources)
    if grok_failed:
        return False, support, "grok_error"
    if remaining_budget <= 0:
        return False, support, "budget_exhausted"
    if not followup_queries:
        return False, support, "single_query_plan"

    if _is_multifacet_search_query(query):
        enough_sources = support["sources_count"] >= _MIN_SOURCES_FOR_EARLY_STOP_COMPLEX
        enough_domains = support["domain_count"] >= _MIN_DOMAIN_COUNT_FOR_EARLY_STOP_COMPLEX
        enough_claims = (
            support["claim_count"] == 0
            or support["supported_claims"]
            >= min(support["claim_count"], _MIN_SUPPORTED_CLAIMS_FOR_EARLY_STOP_COMPLEX)
        )
    else:
        enough_sources = support["sources_count"] >= _MIN_SOURCES_FOR_EARLY_STOP
        enough_domains = support["domain_count"] >= _MIN_DOMAIN_COUNT_FOR_EARLY_STOP
        enough_claims = (
            support["claim_count"] == 0
            or support["supported_claims"] >= min(support["claim_count"], 2)
        )
    if enough_sources and enough_domains and enough_claims:
        return False, support, "initial_support_sufficient"
    return True, support, "initial_support_insufficient"


async def _safe_tavily_search(search_query: str, count: int) -> list[dict] | None:
    try:
        if count:
            return await _call_tavily_search(search_query, count)
    except Exception:
        return None
    return None


async def _safe_firecrawl_search(search_query: str, count: int) -> list[dict] | None:
    try:
        if count:
            return await _call_firecrawl_search(search_query, count)
    except Exception:
        return None
    return None


def _external_search_timeout(phase: str, *, relaxed: bool = False) -> float:
    if relaxed:
        return _RELAXED_EXTERNAL_SEARCH_TIMEOUT_SECONDS
    if phase == "expansion":
        return _EXPANSION_EXTERNAL_SEARCH_TIMEOUT_SECONDS
    return _EXTERNAL_SEARCH_TIMEOUT_SECONDS


async def _run_external_search_specs(
    task_specs: list[dict[str, Any]],
    *,
    phase: str,
    relaxed: bool = False,
) -> list[tuple[dict[str, Any], list[dict] | None]]:
    coros: list[Any] = []
    active_specs: list[dict[str, Any]] = []
    timeout_seconds = _external_search_timeout(phase, relaxed=relaxed)

    for spec in task_specs:
        provider = spec.get("provider")
        search_query = str(spec.get("query") or "").strip()
        count = int(spec.get("requested") or 0)
        if not search_query or count <= 0:
            continue
        active_specs.append(spec)
        if provider == "tavily":
            coros.append(asyncio.wait_for(_safe_tavily_search(search_query, count), timeout=timeout_seconds))
        else:
            coros.append(asyncio.wait_for(_safe_firecrawl_search(search_query, count), timeout=timeout_seconds))

    if not coros:
        return []

    gathered = await asyncio.gather(*coros, return_exceptions=True)
    results: list[tuple[dict[str, Any], list[dict] | None]] = []
    for spec, result in zip(active_specs, gathered):
        if isinstance(result, asyncio.TimeoutError):
            result_list = None
        else:
            result_list = result if isinstance(result, list) else None
        results.append((spec, result_list))
    return results


async def _retry_zero_result_queries(
    task_results: list[tuple[dict[str, Any], list[dict] | None]],
    *,
    phase: str,
) -> list[tuple[dict[str, Any], list[dict] | None]]:
    if phase != "expansion" or not task_results:
        return task_results

    grouped: dict[str, list[int]] = {}
    for index, (spec, _result_list) in enumerate(task_results):
        grouped.setdefault(str(spec.get("query") or ""), []).append(index)

    relaxed_specs: list[dict[str, Any]] = []
    relaxed_targets: list[int] = []
    for original_query, indexes in grouped.items():
        total_results = sum(len(task_results[index][1] or []) for index in indexes)
        if total_results > 0:
            continue
        relaxed_query = _build_relaxed_search_query(original_query)
        if not relaxed_query or relaxed_query.casefold() == original_query.casefold():
            continue
        for index in indexes:
            spec, _result_list = task_results[index]
            relaxed_specs.append({**spec, "query": relaxed_query, "fallback_from": original_query})
            relaxed_targets.append(index)

    if not relaxed_specs:
        return task_results

    rerun_results = await _run_external_search_specs(relaxed_specs, phase=phase, relaxed=True)
    if not rerun_results:
        return task_results

    updated = list(task_results)
    for target_index, rerun in zip(relaxed_targets, rerun_results):
        updated[target_index] = rerun
    return updated


async def _run_external_search_batch(
    queries: list[str],
    tavily_total: int,
    firecrawl_total: int,
    *,
    phase: str,
) -> tuple[list[dict], list[list[dict]], list[list[dict]], int]:
    task_specs: list[dict[str, Any]] = []
    tavily_budgets = _allocate_query_budgets(tavily_total, len(queries))
    firecrawl_budgets = _allocate_query_budgets(firecrawl_total, len(queries))

    for search_query, count in zip(queries, tavily_budgets):
        if count <= 0:
            continue
        task_specs.append(
            {"provider": "tavily", "query": search_query, "requested": count, "phase": phase}
        )

    for search_query, count in zip(queries, firecrawl_budgets):
        if count <= 0:
            continue
        task_specs.append(
            {"provider": "firecrawl", "query": search_query, "requested": count, "phase": phase}
        )

    if not task_specs:
        return [], [], [], 0

    task_results = await _run_external_search_specs(task_specs, phase=phase)
    task_results = await _retry_zero_result_queries(task_results, phase=phase)

    records: list[dict] = []
    tavily_source_chunks: list[list[dict]] = []
    firecrawl_source_chunks: list[list[dict]] = []
    used_budget = 0

    for spec, result_list in task_results:
        used_budget += int(spec["requested"])
        record = {
            "provider": spec["provider"],
            "query": spec["query"],
            "requested": spec["requested"],
            "results_count": len(result_list or []),
            "phase": phase,
        }
        if spec.get("fallback_from"):
            record["fallback_from"] = spec["fallback_from"]
        records.append(record)
        if spec["provider"] == "tavily":
            tavily_source_chunks.append(
                _extra_results_to_sources(result_list, None, query_used=spec["query"])
            )
        else:
            firecrawl_source_chunks.append(
                _extra_results_to_sources(None, result_list, query_used=spec["query"])
            )

    return records, tavily_source_chunks, firecrawl_source_chunks, used_budget


def _planning_session_or_error(session_id: str):
    session = planning_engine.get_session(session_id)
    if not session:
        return None, {"error": f"Session '{session_id}' not found. Call plan_intent first."}
    return session, None


def _validate_model_or_error(model_cls, data: dict) -> tuple[dict | None, str | None]:
    try:
        validated = model_cls.model_validate(data)
    except ValidationError as exc:
        return None, _validation_error("输入参数不合法", exc.errors())
    return validated.model_dump(exclude_none=True), None


def _extra_results_to_sources(
    tavily_results: list[dict] | None,
    firecrawl_results: list[dict] | None,
    query_used: str = "",
) -> list[dict]:
    sources: list[dict] = []
    seen: set[str] = set()

    if firecrawl_results:
        for r in firecrawl_results:
            url = (r.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            item: dict = {"url": url, "provider": "firecrawl"}
            if query_used:
                item["query_used"] = query_used
            title = (r.get("title") or "").strip()
            if title:
                item["title"] = title
            desc = (r.get("description") or "").strip()
            if desc:
                item["description"] = desc
            facet = (r.get("facet") or "").strip()
            if facet:
                item["facet"] = facet
            sources.append(item)

    if tavily_results:
        for r in tavily_results:
            url = (r.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            item: dict = {"url": url, "provider": "tavily"}
            if query_used:
                item["query_used"] = query_used
            title = (r.get("title") or "").strip()
            if title:
                item["title"] = title
            content = (r.get("content") or "").strip()
            if content:
                item["description"] = content
            facet = (r.get("facet") or "").strip()
            if facet:
                item["facet"] = facet
            score = r.get("score")
            if isinstance(score, (int, float)):
                item["score"] = score
            sources.append(item)

    return sources


def _source_needs_enrichment(source: dict) -> bool:
    title = (source.get("title") or "").strip()
    description = (source.get("description") or "").strip()
    url = (source.get("url") or "").strip()
    return (not title or title == url) and not description


def _query_count_for_sources(root_query: str, sources: list[dict]) -> int:
    seen: set[str] = set()
    normalized_root = " ".join((root_query or "").split()).strip()
    if normalized_root:
        seen.add(normalized_root.casefold())

    for source in sources:
        query_used = " ".join(((source.get("query_used") or "")).split()).strip()
        if query_used:
            seen.add(query_used.casefold())

    return len(seen)


def _should_enrich_sources(query: str, sources: list[dict]) -> bool:
    if not any(_source_needs_enrichment(source) for source in sources):
        return False

    if len(sources) <= 3:
        return True

    if _query_count_for_sources(query, sources) > 1:
        return True

    return _should_expand_search_query(query, _INITIAL_EXTRA_SOURCE_BUDGET_CAP)


def _should_rank_sources(query: str, sources: list[dict]) -> bool:
    if len(sources) < _MIN_SOURCES_FOR_RANKING:
        return False

    if _is_multifacet_search_query(query):
        return False

    query_count = _query_count_for_sources(query, sources)
    if query_count > 1:
        return True

    return _should_expand_search_query(query, _INITIAL_EXTRA_SOURCE_BUDGET_CAP)


def _truncate_text(value: str, max_len: int = 280) -> str:
    text = (value or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _build_sources_only_fallback_answer(query: str, sources: list[dict], *, reason: str = "") -> str:
    if not sources:
        return f"搜索失败: {reason}" if reason else ""

    lines = ["聚合搜索已完成，但综合生成失败，以下为保底来源摘要：", ""]
    if query:
        lines.append(f"Query: {query}")
        lines.append("")

    for source in sources[:5]:
        title = _truncate_text(source.get("title") or source.get("url") or "Untitled", 120)
        description = _truncate_text(source.get("description") or "", 220)
        url = (source.get("url") or "").strip()
        line = f"- **{title}**"
        if description:
            line += f": {description}"
        if url:
            line += f" ({url})"
        lines.append(line)

    if reason:
        lines.extend(["", f"降级原因: {reason}"])
    lines.extend(["", "以上为保底结果，建议稍后重试以获取完整综合答案。"])
    return "\n".join(lines)


def _build_rank_sources_text(sources: list[dict]) -> str:
    lines: list[str] = []
    for idx, source in enumerate(sources, 1):
        title = _truncate_text(source.get("title") or source.get("url") or "Untitled", 120)
        url = (source.get("url") or "").strip()
        description = _truncate_text(source.get("description") or "", 280)
        lines.append(f"{idx}. Title: {title}")
        if url:
            lines.append(f"   URL: {url}")
        if description:
            lines.append(f"   Summary: {description}")
    return "\n".join(lines)


def _build_synthesis_sources_text(sources: list[dict]) -> str:
    lines: list[str] = []
    for idx, source in enumerate(sources[:_MAX_SYNTHESIS_SOURCES], 1):
        title = _truncate_text(source.get("title") or source.get("url") or "Untitled", 120)
        url = (source.get("url") or "").strip()
        description = _truncate_text(source.get("description") or "", 360)
        provider = (source.get("provider") or "").strip()
        query_used = (source.get("query_used") or "").strip()
        lines.append(f"[{idx}] {title}")
        if url:
            lines.append(f"URL: {url}")
            lines.append(f"Citation: [[{idx}]]({url})")
        if provider:
            lines.append(f"Provider: {provider}")
        if query_used:
            lines.append(f"Query Used: {query_used}")
        if description:
            lines.append(f"Summary: {description}")
        lines.append("")
    return "\n".join(lines).strip()


async def _synthesize_answer_from_sources(
    query: str,
    grok_provider: GrokSearchProvider,
    sources: list[dict],
    fallback_provider: GrokSearchProvider | None = None,
) -> str:
    sources_text = _build_synthesis_sources_text(sources)
    if not sources_text.strip():
        return ""
    try:
        return await asyncio.wait_for(
            grok_provider.synthesize_from_sources(query, sources_text),
            timeout=_SOURCE_SYNTHESIS_TIMEOUT_SECONDS,
        )
    except (Exception, asyncio.TimeoutError):
        if fallback_provider and fallback_provider is not grok_provider:
            try:
                return await asyncio.wait_for(
                    fallback_provider.synthesize_from_sources(query, sources_text),
                    timeout=_SOURCE_SYNTHESIS_TIMEOUT_SECONDS,
                )
            except (Exception, asyncio.TimeoutError):
                return ""
        return ""


def _apply_source_order(sources: list[dict], order: list[int]) -> list[dict]:
    if not sources:
        return []

    ordered: list[dict] = []
    seen: set[int] = set()
    for index in order:
        zero_based = index - 1
        if 0 <= zero_based < len(sources) and zero_based not in seen:
            ordered.append(sources[zero_based])
            seen.add(zero_based)

    for index, source in enumerate(sources):
        if index not in seen:
            ordered.append(source)

    return ordered


async def _enrich_and_rank_sources(
    query: str,
    grok_provider: GrokSearchProvider,
    analysis_provider: GrokSearchProvider | None,
    sources: list[dict],
) -> tuple[list[dict], dict[str, Any]]:
    if not sources:
        return [], {
            "enrichment_considered": False,
            "enrichment_applied": False,
            "ranking_considered": False,
            "ranking_applied": False,
            "ranked_source_count": 0,
        }

    prepared = [dict(item) for item in sources]
    provider_for_analysis = analysis_provider or grok_provider
    allow_enrichment = _should_enrich_sources(query, prepared)
    allow_ranking = _should_rank_sources(query, prepared)
    metadata: dict[str, Any] = {
        "enrichment_considered": allow_enrichment,
        "enrichment_applied": False,
        "ranking_considered": allow_ranking,
        "ranking_applied": False,
        "ranked_source_count": 0,
        "local_priority_considered": _should_prioritize_sources_locally(query, prepared),
        "local_priority_applied": False,
    }

    enrichable = [
        (index, source["url"])
        for index, source in enumerate(prepared)
        if source.get("url") and _source_needs_enrichment(source)
    ][:_MAX_ENRICHABLE_SOURCES]

    if allow_enrichment and enrichable:
        try:
            describe_results = await asyncio.wait_for(
                asyncio.gather(
                    *[provider_for_analysis.describe_url(url) for _, url in enrichable],
                    return_exceptions=True,
                ),
                timeout=_SOURCE_ENRICH_TIMEOUT_SECONDS,
            )
        except (Exception, asyncio.TimeoutError):
            if provider_for_analysis is not grok_provider:
                try:
                    describe_results = await asyncio.wait_for(
                        asyncio.gather(
                            *[grok_provider.describe_url(url) for _, url in enrichable],
                            return_exceptions=True,
                        ),
                        timeout=_SOURCE_ENRICH_TIMEOUT_SECONDS,
                    )
                except (Exception, asyncio.TimeoutError):
                    describe_results = []
            else:
                describe_results = []
        for (index, _url), result in zip(enrichable, describe_results):
            if isinstance(result, Exception) or not isinstance(result, dict):
                continue
            title = (result.get("title") or "").strip()
            extracts = (result.get("extracts") or "").strip()
            if title:
                prepared[index]["title"] = title
            if extracts:
                prepared[index]["description"] = extracts
                metadata["enrichment_applied"] = True

    prioritized_sources, priority_changed = _prioritize_sources_locally(query, prepared)
    if priority_changed:
        prepared = prioritized_sources
        metadata["local_priority_applied"] = True

    if len(prepared) <= 1:
        return prepared, metadata

    if not allow_ranking:
        return prepared, metadata

    rank_candidates = prepared[:_MAX_RANKABLE_SOURCES]
    sources_text = _build_rank_sources_text(rank_candidates)
    if not sources_text.strip():
        return prepared, metadata

    try:
        order = await asyncio.wait_for(
            provider_for_analysis.rank_sources(query, sources_text, len(rank_candidates)),
            timeout=_SOURCE_RANK_TIMEOUT_SECONDS,
        )
    except (Exception, asyncio.TimeoutError):
        if provider_for_analysis is not grok_provider:
            try:
                order = await asyncio.wait_for(
                    grok_provider.rank_sources(query, sources_text, len(rank_candidates)),
                    timeout=_SOURCE_RANK_TIMEOUT_SECONDS,
                )
            except (Exception, asyncio.TimeoutError):
                return prepared, metadata
        else:
            return prepared, metadata

    metadata["ranking_applied"] = True
    metadata["ranked_source_count"] = len(rank_candidates)
    prepared = ranked_candidates = _apply_source_order(rank_candidates, order) + prepared[_MAX_RANKABLE_SOURCES:]
    reprioritized_sources, reprioritized_changed = _prioritize_sources_locally(query, prepared)
    if reprioritized_changed:
        prepared = reprioritized_sources
        metadata["local_priority_applied"] = True
    return prepared, metadata


@mcp.tool(
    name="web_search",
    output_schema=None,
    description="""
    Before using this tool, please use the plan_intent tool to plan the search carefully.
    Performs a deep web search based on the given query and returns Grok's answer directly.

    This tool extracts sources if provided by upstream, caches them, and returns:
    - session_id: string (When you feel confused or curious about the main content, use this field to invoke the get_sources tool to obtain the corresponding list of information sources)
    - content: string (answer only)
    - sources_count: int
    Additional structured search metadata is available through get_sources.
    """,
    meta={"version": "2.0.0", "author": "guda.studio"},
)
async def web_search(
    query: Annotated[str, "Clear, self-contained natural-language search query."],
    platform: Annotated[str, "Target platform to focus on (e.g., 'Twitter', 'GitHub', 'Reddit'). Leave empty for general web search."] = "",
    model: Annotated[str, "Optional model ID for this request only. This value is used ONLY when user explicitly provided."] = "",
    extra_sources: Annotated[int, "Number of additional reference results from Tavily/Firecrawl. Set 0 to disable. Default 20."] = 20,
) -> dict:
    session_id = new_session_id()
    try:
        api_url = config.grok_api_url
        api_key = config.grok_api_key
    except ValueError as e:
        await _SOURCES_CACHE.set(session_id, [])
        return {"session_id": session_id, "content": f"配置错误: {str(e)}", "sources_count": 0}

    effective_model = config.grok_model
    if model:
        available = await _get_available_models_cached(api_url, api_key)
        if available and model not in available:
            await _SOURCES_CACHE.set(session_id, [])
            return {"session_id": session_id, "content": f"无效模型: {model}", "sources_count": 0}
        effective_model = model

    planned_queries = _build_search_queries(query, extra_sources)
    executed_queries: list[dict] = []
    phase_summaries: list[dict[str, Any]] = []
    prefer_source_synthesis = _is_multifacet_search_query(query)
    stage_models = await _resolve_stage_models(
        api_url,
        api_key,
        effective_model,
        query=query,
        prefer_source_synthesis=prefer_source_synthesis,
        planned_queries=planned_queries,
    )
    grok_provider = GrokSearchProvider(api_url, api_key, stage_models["search"])
    analysis_provider = (
        GrokSearchProvider(api_url, api_key, stage_models["analysis"])
        if stage_models["analysis"] != stage_models["search"]
        else grok_provider
    )

    # 计算额外信源配额
    has_tavily = config.tavily_enabled and bool(config.tavily_api_key)
    has_firecrawl = bool(config.firecrawl_api_key)
    initial_budget = _select_initial_extra_source_budget(extra_sources, planned_queries)
    initial_queries = planned_queries[:1]
    followup_queries = planned_queries[1:]
    expansion_budget = _select_expansion_extra_source_budget(
        query,
        max(extra_sources - initial_budget, 0),
        followup_queries,
    )
    initial_tavily_count, initial_firecrawl_count = _split_extra_sources_budget(
        initial_budget, has_tavily, has_firecrawl
    )

    # 首轮执行：简单查询保留主搜索并行，复杂查询优先做外部聚合后快速综合
    grok_error: str | None = None
    grok_fallback_attempted = False
    grok_fallback_applied = False
    grok_fallback_reason = ""

    async def _safe_grok() -> str:
        nonlocal grok_error
        try:
            return await grok_provider.search(query, platform)
        except Exception as exc:
            grok_error = str(exc)
            return ""

    async def _run_grok_fallback_search(trigger_reason: str) -> tuple[str, list[dict]]:
        nonlocal grok_error, grok_fallback_attempted, grok_fallback_applied, grok_fallback_reason
        grok_fallback_attempted = True
        grok_fallback_reason = trigger_reason
        try:
            fallback_result = await grok_provider.search(query, platform)
            fallback_answer, fallback_sources = split_answer_and_sources(fallback_result)
            executed_queries.append(
                {
                    "provider": "grok",
                    "query": query,
                    "requested": 1,
                    "results_count": len(fallback_sources),
                    "status": "ok" if (fallback_answer or fallback_sources) else "empty",
                    "phase": "fallback",
                    "reason": trigger_reason,
                }
            )
            if fallback_answer:
                grok_fallback_applied = True
            return fallback_answer, fallback_sources
        except Exception as exc:
            grok_error = str(exc)
            executed_queries.append(
                {
                    "provider": "grok",
                    "query": query,
                    "requested": 1,
                    "results_count": 0,
                    "status": "error",
                    "phase": "fallback",
                    "reason": trigger_reason,
                }
            )
            return "", []

    if prefer_source_synthesis:
        grok_result = ""
        initial_batch = await _run_external_search_batch(
            initial_queries,
            initial_tavily_count,
            initial_firecrawl_count,
            phase="initial",
        )
    else:
        grok_result, initial_batch = await asyncio.gather(
            _safe_grok(),
            _run_external_search_batch(
                initial_queries,
                initial_tavily_count,
                initial_firecrawl_count,
                phase="initial",
            ),
        )
    initial_records, initial_tavily_chunks, initial_firecrawl_chunks, initial_used_budget = initial_batch
    executed_queries.extend(initial_records)

    answer, grok_sources = split_answer_and_sources(grok_result)
    executed_queries.insert(
        0,
        {
            "provider": "grok",
            "query": query,
            "requested": 1,
            "results_count": len(grok_sources),
            "status": "skipped" if prefer_source_synthesis else ("error" if grok_error else "ok"),
            "phase": "initial",
        },
    )
    initial_sources = merge_sources(
        grok_sources,
        *initial_firecrawl_chunks,
        *initial_tavily_chunks,
    )
    phase_summaries.append(
        {
            "name": "initial",
            "planned_queries": initial_queries,
            "budget_requested": initial_budget,
            "budget_used": initial_used_budget,
            "task_count": len(initial_records),
            "source_count": len(initial_sources),
            "skipped": False,
        }
    )

    should_expand, initial_support, decision_reason = _should_expand_after_initial(
        query,
        answer,
        initial_sources,
        followup_queries,
        expansion_budget,
        grok_failed=bool(grok_error and not answer),
    )

    expansion_used_budget = 0
    expansion_records: list[dict] = []
    expansion_tavily_chunks: list[list[dict]] = []
    expansion_firecrawl_chunks: list[list[dict]] = []
    expansion_queries = followup_queries
    if _is_multifacet_search_query(query):
        expansion_queries = followup_queries[:_MAX_FOLLOWUP_QUERIES_FOR_MULTIFACET]
    if should_expand and expansion_queries:
        expansion_tavily_count, expansion_firecrawl_count = _split_extra_sources_budget(
            expansion_budget, has_tavily, has_firecrawl
        )
        expansion_records, expansion_tavily_chunks, expansion_firecrawl_chunks, expansion_used_budget = (
            await _run_external_search_batch(
                expansion_queries,
                expansion_tavily_count,
                expansion_firecrawl_count,
                phase="expansion",
            )
        )
        executed_queries.extend(expansion_records)
        expansion_sources = merge_sources(*expansion_firecrawl_chunks, *expansion_tavily_chunks)
        phase_summaries.append(
            {
                "name": "expansion",
                "planned_queries": expansion_queries,
                "budget_requested": expansion_budget,
                "budget_used": expansion_used_budget,
                "task_count": len(expansion_records),
                "source_count": len(expansion_sources),
                "skipped": False,
                "reason": decision_reason,
            }
        )
        all_sources = merge_sources(initial_sources, *expansion_firecrawl_chunks, *expansion_tavily_chunks)
    else:
        phase_summaries.append(
            {
                "name": "expansion",
                "planned_queries": expansion_queries,
                "budget_requested": expansion_budget,
                "budget_used": 0,
                "task_count": 0,
                "source_count": 0,
                "skipped": True,
                "reason": decision_reason,
            }
        )
        all_sources = initial_sources

    postprocessing: dict[str, Any] = {}
    if all_sources:
        all_sources, postprocessing = await _enrich_and_rank_sources(
            query,
            grok_provider,
            analysis_provider,
            all_sources,
        )

    synthesis_reason = ""
    if prefer_source_synthesis and all_sources:
        answer = await _synthesize_answer_from_sources(query, analysis_provider, all_sources, grok_provider)
        if not answer:
            synthesis_reason = "source_synthesis_failed"
    if prefer_source_synthesis and not answer:
        fallback_reason = "source_synthesis_failed" if all_sources else "no_sources_after_external_search"
        fallback_answer, fallback_sources = await _run_grok_fallback_search(fallback_reason)
        if fallback_sources:
            all_sources = merge_sources(all_sources, fallback_sources)
        if fallback_answer:
            answer = fallback_answer
    elif not answer and grok_error:
        if all_sources:
            answer = await _synthesize_answer_from_sources(query, analysis_provider, all_sources, grok_provider)
            if not answer:
                synthesis_reason = grok_error
        else:
            answer = f"搜索失败: {grok_error}"

    if not answer and all_sources:
        answer = _build_sources_only_fallback_answer(query, all_sources, reason=synthesis_reason or grok_error or "synthesis_unavailable")
    elif not answer:
        answer = f"搜索失败: {grok_error or grok_fallback_reason or 'no_results'}"

    inline_citation_sources = _sources_from_inline_citations(answer)
    if inline_citation_sources:
        all_sources = merge_sources(all_sources, inline_citation_sources)

    answer, evidence_bindings = _attach_evidence_citations(answer, all_sources)

    search_trace = _build_search_trace(
        query,
        planned_queries,
        executed_queries,
        all_sources,
        requested_budget=extra_sources,
        used_budget=initial_used_budget + expansion_used_budget,
        phases=phase_summaries,
        decision={
            "early_stopped": bool(followup_queries and not should_expand),
            "reason": decision_reason,
            "initial_support": initial_support,
        },
        postprocessing={
            **postprocessing,
            "search_model": stage_models["search"],
            "analysis_model": stage_models["analysis"],
            "grok_fallback_attempted": grok_fallback_attempted,
            "grok_fallback_applied": grok_fallback_applied,
            "grok_fallback_reason": grok_fallback_reason,
            "source_synthesis_preferred": prefer_source_synthesis,
            "source_synthesis_applied": bool(
                prefer_source_synthesis
                and all_sources
                and answer
                and not synthesis_reason
                and not grok_fallback_applied
            ),
            "source_synthesis_reason": synthesis_reason,
            "inline_citation_sources_count": len(inline_citation_sources),
        },
    )
    await _SOURCES_CACHE.set(
        session_id,
        {
            "sources": all_sources,
            "search_trace": search_trace,
            "evidence_bindings": evidence_bindings,
        },
    )
    return {"session_id": session_id, "content": answer, "sources_count": len(all_sources)}


@mcp.tool(
    name="get_sources",
    description="""
    When you feel confused or curious about the search response content, use the session_id returned by web_search to invoke the this tool to obtain the corresponding list of information sources.
    Retrieve all cached sources for a previous web_search call.
    Provide the session_id returned by web_search to get the full source list.
    When available, the response also includes:
    - search_trace: expanded queries and provider execution details
    - evidence_bindings: answer claims mapped to supporting sources
    """,
    meta={"version": "1.0.0", "author": "guda.studio"},
)
async def get_sources(
    session_id: Annotated[str, "Session ID from previous web_search call."]
) -> dict:
    payload = await _SOURCES_CACHE.get(session_id)
    if payload is None:
        return {
            "session_id": session_id,
            "sources": [],
            "sources_count": 0,
            "error": "session_id_not_found_or_expired",
        }

    normalized = _normalize_cached_search_payload(payload)
    response = {
        "session_id": session_id,
        "sources": normalized["sources"],
        "sources_count": len(normalized["sources"]),
    }
    if "search_trace" in normalized:
        response["search_trace"] = normalized["search_trace"]
    if "evidence_bindings" in normalized:
        response["evidence_bindings"] = normalized["evidence_bindings"]
    return response


async def _call_tavily_extract(url: str) -> str | None:
    import httpx
    api_url = config.tavily_api_url
    api_key = config.tavily_api_key
    if not config.tavily_enabled or not api_key:
        return None
    endpoint = f"{api_url.rstrip('/')}/extract"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"urls": [url], "format": "markdown"}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(endpoint, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
            if data.get("results") and len(data["results"]) > 0:
                content = data["results"][0].get("raw_content", "")
                return content if content and content.strip() else None
            return None
    except Exception:
        return None


async def _call_tavily_search(query: str, max_results: int = 6) -> list[dict] | None:
    import httpx
    api_key = config.tavily_api_key
    if not config.tavily_enabled or not api_key:
        return None
    endpoint = f"{config.tavily_api_url.rstrip('/')}/search"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
        "include_raw_content": False,
        "include_answer": False,
    }
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(endpoint, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            return [
                {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", ""), "score": r.get("score", 0)}
                for r in results
            ] if results else None
    except Exception:
        return None


async def _call_firecrawl_search(query: str, limit: int = 14) -> list[dict] | None:
    import httpx
    api_key = config.firecrawl_api_key
    if not api_key:
        return None
    endpoint = f"{config.firecrawl_api_url.rstrip('/')}/search"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"query": query, "limit": limit}
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(endpoint, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
            results = data.get("data", {}).get("web", [])
            return [
                {"title": r.get("title", ""), "url": r.get("url", ""), "description": r.get("description", "")}
                for r in results
            ] if results else None
    except Exception:
        return None


async def _call_firecrawl_scrape(url: str, ctx=None) -> str | None:
    import httpx
    api_url = config.firecrawl_api_url
    api_key = config.firecrawl_api_key
    if not api_key:
        return None
    endpoint = f"{api_url.rstrip('/')}/scrape"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    max_retries = config.retry_max_attempts
    for attempt in range(max_retries):
        body = {
            "url": url,
            "formats": ["markdown"],
            "timeout": 60000,
            "waitFor": (attempt + 1) * 1500,
        }
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(endpoint, headers=headers, json=body)
                response.raise_for_status()
                data = response.json()
                markdown = data.get("data", {}).get("markdown", "")
                if markdown and markdown.strip():
                    return markdown
                await log_info(ctx, f"Firecrawl: markdown为空, 重试 {attempt + 1}/{max_retries}", config.debug_enabled)
        except Exception as e:
            await log_info(ctx, f"Firecrawl error: {e}", config.debug_enabled)
            return None
    return None


@mcp.tool(
    name="web_fetch",
    output_schema=None,
    description="""
    Fetches and extracts complete content from a URL, returning it as a structured Markdown document.

    **Key Features:**
        - **Full Content Extraction:** Retrieves and parses all meaningful content (text, images, links, tables, code blocks).
        - **Markdown Conversion:** Converts HTML structure to well-formatted Markdown with preserved hierarchy.
        - **Content Fidelity:** Maintains 100% content fidelity without summarization or modification.

    **Edge Cases & Best Practices:**
        - Ensure URL is complete and accessible (not behind authentication or paywalls).
        - May not capture dynamically loaded content requiring JavaScript execution.
        - Large pages may take longer to process; consider timeout implications.
    """,
    meta={"version": "1.3.0", "author": "guda.studio"},
)
async def web_fetch(
    url: Annotated[str, "Valid HTTP/HTTPS web address pointing to the target page. Must be complete and accessible."],
    ctx: Context = None
) -> str:
    if not _is_valid_web_url(url):
        return f"无效URL: {url}"

    await log_info(ctx, f"Begin Fetch: {url}", config.debug_enabled)

    result = await _call_tavily_extract(url)
    if result:
        await log_info(ctx, "Fetch Finished (Tavily)!", config.debug_enabled)
        return result

    await log_info(ctx, "Tavily unavailable or failed, trying Firecrawl...", config.debug_enabled)
    result = await _call_firecrawl_scrape(url, ctx)
    if result:
        await log_info(ctx, "Fetch Finished (Firecrawl)!", config.debug_enabled)
        return result

    await log_info(ctx, "Fetch Failed!", config.debug_enabled)
    if not config.tavily_api_key and not config.firecrawl_api_key:
        return "配置错误: TAVILY_API_KEY 和 FIRECRAWL_API_KEY 均未配置"
    return "提取失败: 所有提取服务均未能获取内容"


async def _call_tavily_map(url: str, instructions: str = None, max_depth: int = 1,
                           max_breadth: int = 20, limit: int = 50, timeout: int = 150) -> str:
    import httpx
    import json
    api_url = config.tavily_api_url
    api_key = config.tavily_api_key
    if not config.tavily_enabled or not api_key:
        return "配置错误: TAVILY_API_KEY 未配置，请设置环境变量 TAVILY_API_KEY"
    endpoint = f"{api_url.rstrip('/')}/map"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"url": url, "max_depth": max_depth, "max_breadth": max_breadth, "limit": limit, "timeout": timeout}
    if instructions:
        body["instructions"] = instructions
    try:
        async with httpx.AsyncClient(timeout=float(timeout + 10)) as client:
            response = await client.post(endpoint, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
            return json.dumps({
                "base_url": data.get("base_url", ""),
                "results": data.get("results", []),
                "response_time": data.get("response_time", 0)
            }, ensure_ascii=False, indent=2)
    except httpx.TimeoutException:
        return f"映射超时: 请求超过{timeout}秒"
    except httpx.HTTPStatusError as e:
        return f"HTTP错误: {e.response.status_code} - {e.response.text[:200]}"
    except Exception as e:
        return f"映射错误: {str(e)}"


@mcp.tool(
    name="web_map",
    description="""
    Maps a website's structure by traversing it like a graph, discovering URLs and generating a comprehensive site map.

    **Key Features:**
        - **Graph Traversal:** Explores website structure starting from root URL.
        - **Depth & Breadth Control:** Configure traversal limits to balance coverage and performance.
        - **Instruction Filtering:** Use natural language to focus crawler on specific content types.

    **Edge Cases & Best Practices:**
        - Start with low max_depth (1-2) for initial exploration, increase if needed.
        - Use instructions to filter for specific content (e.g., "only documentation pages").
        - Large sites may hit timeout limits; adjust timeout and limit parameters accordingly.
    """,
    meta={"version": "1.3.0", "author": "guda.studio"},
)
async def web_map(
    url: Annotated[str, "Root URL to begin the mapping (e.g., 'https://docs.example.com')."],
    instructions: Annotated[str, "Natural language instructions for the crawler to filter or focus on specific content."] = "",
    max_depth: Annotated[int, Field(description="Maximum depth of mapping from the base URL.", ge=1, le=5)] = 1,
    max_breadth: Annotated[int, Field(description="Maximum number of links to follow per page.", ge=1, le=500)] = 20,
    limit: Annotated[int, Field(description="Total number of links to process before stopping.", ge=1, le=500)] = 50,
    timeout: Annotated[int, Field(description="Maximum time in seconds for the operation.", ge=10, le=150)] = 150
) -> str:
    if not _is_valid_web_url(url):
        return f"无效URL: {url}"
    result = await _call_tavily_map(url, instructions, max_depth, max_breadth, limit, timeout)
    return result


@mcp.tool(
    name="get_config_info",
    output_schema=None,
    description="""
    Returns current Grok Search MCP server configuration and tests API connectivity.

    **Key Features:**
        - **Configuration Check:** Verifies environment variables and current settings.
        - **Connection Test:** Sends request to /models endpoint to validate API access.
        - **Model Discovery:** Lists all available models from the API.

    **Edge Cases & Best Practices:**
        - Use this tool first when debugging connection or configuration issues.
        - API keys are automatically masked for security in the response.
        - Connection test timeout is 10 seconds; network issues may cause delays.
    """,
    meta={"version": "1.3.0", "author": "guda.studio"},
)
async def get_config_info() -> str:
    import json
    import httpx

    config_info = config.get_config_info()

    # 添加连接测试
    test_result = {
        "status": "未测试",
        "message": "",
        "response_time_ms": 0
    }

    try:
        api_url = config.grok_api_url
        api_key = config.grok_api_key

        # 构建 /models 端点 URL
        models_url = f"{api_url.rstrip('/')}/models"

        # 发送测试请求
        import time
        start_time = time.time()

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                models_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
            )

            response_time = (time.time() - start_time) * 1000  # 转换为毫秒

            if response.status_code == 200:
                test_result["status"] = "✅ 连接成功"
                test_result["message"] = f"成功获取模型列表 (HTTP {response.status_code})"
                test_result["response_time_ms"] = round(response_time, 2)

                # 尝试解析返回的模型列表
                try:
                    models_data = response.json()
                    if "data" in models_data and isinstance(models_data["data"], list):
                        model_count = len(models_data["data"])
                        test_result["message"] += f"，共 {model_count} 个模型"

                        # 提取所有模型的 ID/名称
                        model_names = []
                        for model in models_data["data"]:
                            if isinstance(model, dict) and "id" in model:
                                model_names.append(model["id"])

                        if model_names:
                            test_result["available_models"] = model_names
                except:
                    pass
            else:
                test_result["status"] = "⚠️ 连接异常"
                test_result["message"] = f"HTTP {response.status_code}: {response.text[:100]}"
                test_result["response_time_ms"] = round(response_time, 2)

    except httpx.TimeoutException:
        test_result["status"] = "❌ 连接超时"
        test_result["message"] = "请求超时（10秒），请检查网络连接或 API URL"
    except httpx.RequestError as e:
        test_result["status"] = "❌ 连接失败"
        test_result["message"] = f"网络错误: {str(e)}"
    except ValueError as e:
        test_result["status"] = "❌ 配置错误"
        test_result["message"] = str(e)
    except Exception as e:
        test_result["status"] = "❌ 测试失败"
        test_result["message"] = f"未知错误: {str(e)}"

    config_info["connection_test"] = test_result

    return json.dumps(config_info, ensure_ascii=False, indent=2)


@mcp.tool(
    name="switch_model",
    output_schema=None,
    description="""
    Switches the default Grok model used for search and fetch operations, persisting the setting.

    **Key Features:**
        - **Model Selection:** Change the AI model for web search and content fetching.
        - **Persistent Storage:** Model preference saved to ~/.config/grok-search/config.json.
        - **Immediate Effect:** New model used for all subsequent operations.

    **Edge Cases & Best Practices:**
        - Use get_config_info to verify available models before switching.
        - Invalid model IDs may cause API errors in subsequent requests.
        - Model changes persist across sessions until explicitly changed again.
    """,
    meta={"version": "1.3.0", "author": "guda.studio"},
)
async def switch_model(
    model: Annotated[str, "Model ID to switch to (e.g., 'grok-4.20-fast', 'grok-4.20-auto', 'grok-4.20-expert')."]
) -> str:
    import json

    try:
        previous_model = config.grok_model
        api_url = config.grok_api_url
        api_key = config.grok_api_key
        available_models = await _get_available_models_cached(api_url, api_key)
        if not available_models:
            result = {
                "status": "❌ 失败",
                "message": "无法获取可用模型列表，未修改当前配置"
            }
            return json.dumps(result, ensure_ascii=False, indent=2)
        if model not in available_models:
            result = {
                "status": "❌ 失败",
                "message": f"无效模型: {model}",
                "available_models": available_models,
            }
            return json.dumps(result, ensure_ascii=False, indent=2)
        config.set_model(model)
        current_model = config.grok_model

        result = {
            "status": "✅ 成功",
            "previous_model": previous_model,
            "current_model": current_model,
            "message": f"模型已从 {previous_model} 切换到 {current_model}",
            "config_file": str(config.config_file),
            "available_models": available_models,
        }

        return json.dumps(result, ensure_ascii=False, indent=2)

    except ValueError as e:
        result = {
            "status": "❌ 失败",
            "message": f"切换模型失败: {str(e)}"
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        result = {
            "status": "❌ 失败",
            "message": f"未知错误: {str(e)}"
        }
        return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool(
    name="toggle_builtin_tools",
    output_schema=None,
    description="""
    Toggle Claude Code's built-in WebSearch and WebFetch tools on/off.

    **Key Features:**
        - **Tool Control:** Enable or disable Claude Code's native web tools.
        - **Project Scope:** Changes apply to current project's .claude/settings.json.
        - **Status Check:** Query current state without making changes.

    **Edge Cases & Best Practices:**
        - Use "on" to block built-in tools when preferring this MCP server's implementation.
        - Use "off" to restore Claude Code's native tools.
        - Use "status" to check current configuration without modification.
    """,
    meta={"version": "1.3.0", "author": "guda.studio"},
)
async def toggle_builtin_tools(
    action: Annotated[str, "Action to perform: 'on' (block built-in), 'off' (allow built-in), or 'status' (check current state)."] = "status"
) -> str:
    import json

    root = _find_project_root(Path.cwd())
    settings_path = root / ".claude" / "settings.json"
    tools = ["WebFetch", "WebSearch"]

    # Load or initialize
    if settings_path.exists():
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
    else:
        settings = {"permissions": {"deny": []}}

    deny = settings.setdefault("permissions", {}).setdefault("deny", [])
    blocked = all(t in deny for t in tools)

    # Execute action
    if action in ["on", "enable"]:
        for t in tools:
            if t not in deny:
                deny.append(t)
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        msg = "官方工具已禁用"
        blocked = True
    elif action in ["off", "disable"]:
        deny[:] = [t for t in deny if t not in tools]
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        msg = "官方工具已启用"
        blocked = False
    else:
        msg = f"官方工具当前{'已禁用' if blocked else '已启用'}"

    return json.dumps({
        "blocked": blocked,
        "deny_list": deny,
        "file": str(settings_path),
        "message": msg
    }, ensure_ascii=False, indent=2)


@mcp.tool(
    name="plan_intent",
    output_schema=None,
    description="""
    Phase 1 of search planning: Analyze user intent. Call this FIRST to create a session.
    Returns session_id for subsequent phases. Required flow:
    plan_intent → plan_complexity → plan_sub_query(×N) → plan_search_term(×N) → plan_tool_mapping(×N) → plan_execution

    Required phases depend on complexity: Level 1 = phases 1-3; Level 2 = phases 1-5; Level 3 = all 6.
    """,
)
async def plan_intent(
    thought: Annotated[str, "Reasoning for this phase"],
    core_question: Annotated[str, "Distilled core question in one sentence"],
    query_type: Annotated[str, "factual | comparative | exploratory | analytical"],
    time_sensitivity: Annotated[str, "realtime | recent | historical | irrelevant"],
    session_id: Annotated[str, "Empty for new session, or existing ID to revise"] = "",
    confidence: Annotated[float, "Confidence 0.0-1.0"] = 1.0,
    domain: Annotated[str, "Specific domain if identifiable"] = "",
    premise_valid: Annotated[Optional[bool], "False if the question contains a flawed assumption"] = None,
    ambiguities: Annotated[str, "Comma-separated unresolved ambiguities"] = "",
    unverified_terms: Annotated[str, "Comma-separated external terms to verify"] = "",
    is_revision: Annotated[bool, "True to overwrite existing intent"] = False,
) -> str:
    import json
    data = {"core_question": core_question, "query_type": query_type, "time_sensitivity": time_sensitivity}
    if domain:
        data["domain"] = domain
    if premise_valid is not None:
        data["premise_valid"] = premise_valid
    if ambiguities:
        data["ambiguities"] = _split_csv(ambiguities)
    if unverified_terms:
        data["unverified_terms"] = _split_csv(unverified_terms)
    data, error = _validate_model_or_error(IntentOutput, data)
    if error:
        return error
    return json.dumps(planning_engine.process_phase(
        phase="intent_analysis", thought=thought, session_id=session_id,
        is_revision=is_revision, confidence=confidence, phase_data=data,
    ), ensure_ascii=False, indent=2)


@mcp.tool(
    name="plan_complexity",
    output_schema=None,
    description="Phase 2: Assess search complexity (1-3). Controls required phases: Level 1 = phases 1-3; Level 2 = phases 1-5; Level 3 = all 6.",
)
async def plan_complexity(
    session_id: Annotated[str, "Session ID from plan_intent"],
    thought: Annotated[str, "Reasoning for complexity assessment"],
    level: Annotated[int, "Complexity 1-3"],
    estimated_sub_queries: Annotated[int, "Expected number of sub-queries"],
    estimated_tool_calls: Annotated[int, "Expected total tool calls"],
    justification: Annotated[str, "Why this complexity level"],
    confidence: Annotated[float, "Confidence 0.0-1.0"] = 1.0,
    is_revision: Annotated[bool, "True to overwrite"] = False,
) -> str:
    import json
    _, session_error = _planning_session_or_error(session_id)
    if session_error:
        return json.dumps(session_error)
    phase_data, error = _validate_model_or_error(
        ComplexityOutput,
        {
            "level": level,
            "estimated_sub_queries": estimated_sub_queries,
            "estimated_tool_calls": estimated_tool_calls,
            "justification": justification,
        },
    )
    if error:
        return error
    return json.dumps(planning_engine.process_phase(
        phase="complexity_assessment", thought=thought, session_id=session_id,
        is_revision=is_revision, confidence=confidence,
        phase_data=phase_data,
    ), ensure_ascii=False, indent=2)


@mcp.tool(
    name="plan_sub_query",
    output_schema=None,
    description="Phase 3: Add one sub-query. Call once per sub-query; data accumulates across calls. Set is_revision=true to replace all.",
)
async def plan_sub_query(
    session_id: Annotated[str, "Session ID from plan_intent"],
    thought: Annotated[str, "Reasoning for this sub-query"],
    id: Annotated[str, "Unique ID (e.g., 'sq1')"],
    goal: Annotated[str, "Sub-query goal"],
    expected_output: Annotated[str, "What success looks like"],
    boundary: Annotated[str, "What this excludes — mutual exclusion with siblings"],
    confidence: Annotated[float, "Confidence 0.0-1.0"] = 1.0,
    depends_on: Annotated[str, "Comma-separated prerequisite IDs"] = "",
    tool_hint: Annotated[str, "web_search | web_fetch | web_map"] = "",
    is_revision: Annotated[bool, "True to replace all sub-queries"] = False,
) -> str:
    import json
    _, session_error = _planning_session_or_error(session_id)
    if session_error:
        return json.dumps(session_error)
    item = {"id": id, "goal": goal, "expected_output": expected_output, "boundary": boundary}
    if depends_on:
        item["depends_on"] = _split_csv(depends_on)
    if tool_hint:
        item["tool_hint"] = tool_hint
    item, error = _validate_model_or_error(SubQuery, item)
    if error:
        return error
    return json.dumps(planning_engine.process_phase(
        phase="query_decomposition", thought=thought, session_id=session_id,
        is_revision=is_revision, confidence=confidence, phase_data=item,
    ), ensure_ascii=False, indent=2)


@mcp.tool(
    name="plan_search_term",
    output_schema=None,
    description="Phase 4: Add one search term. Call once per term; data accumulates. First call must set approach.",
)
async def plan_search_term(
    session_id: Annotated[str, "Session ID from plan_intent"],
    thought: Annotated[str, "Reasoning for this search term"],
    term: Annotated[str, "Search query (max 8 words)"],
    purpose: Annotated[str, "Sub-query ID this serves (e.g., 'sq1')"],
    round: Annotated[int, "Execution round: 1=broad, 2+=targeted follow-up"],
    confidence: Annotated[float, "Confidence 0.0-1.0"] = 1.0,
    approach: Annotated[str, "broad_first | narrow_first | targeted (required on first call)"] = "",
    fallback_plan: Annotated[str, "Fallback if primary searches fail"] = "",
    is_revision: Annotated[bool, "True to replace all search terms"] = False,
) -> str:
    import json
    session, session_error = _planning_session_or_error(session_id)
    if session_error:
        return json.dumps(session_error)
    if _word_count(term) > 8:
        return _validation_error("Search query must be 8 words or fewer")
    if (is_revision or session.phases.get("search_strategy") is None) and not approach:
        return _validation_error("First search term must define approach")
    search_term, error = _validate_model_or_error(
        SearchTerm, {"term": term, "purpose": purpose, "round": round}
    )
    if error:
        return error
    data = {"search_terms": [search_term]}
    if approach:
        data["approach"] = approach
    if fallback_plan:
        data["fallback_plan"] = fallback_plan
    if approach:
        _, error = _validate_model_or_error(StrategyOutput, data)
        if error:
            return error
    return json.dumps(planning_engine.process_phase(
        phase="search_strategy", thought=thought, session_id=session_id,
        is_revision=is_revision, confidence=confidence, phase_data=data,
    ), ensure_ascii=False, indent=2)


@mcp.tool(
    name="plan_tool_mapping",
    output_schema=None,
    description="Phase 5: Map a sub-query to a tool. Call once per mapping; data accumulates.",
)
async def plan_tool_mapping(
    session_id: Annotated[str, "Session ID from plan_intent"],
    thought: Annotated[str, "Reasoning for this mapping"],
    sub_query_id: Annotated[str, "Sub-query ID to map"],
    tool: Annotated[str, "web_search | web_fetch | web_map"],
    reason: Annotated[str, "Why this tool for this sub-query"],
    confidence: Annotated[float, "Confidence 0.0-1.0"] = 1.0,
    params_json: Annotated[str, "Optional JSON string for tool-specific params"] = "",
    is_revision: Annotated[bool, "True to replace all mappings"] = False,
) -> str:
    import json
    _, session_error = _planning_session_or_error(session_id)
    if session_error:
        return json.dumps(session_error)
    item = {"sub_query_id": sub_query_id, "tool": tool, "reason": reason}
    if params_json:
        try:
            item["params"] = json.loads(params_json)
        except json.JSONDecodeError:
            return _validation_error("params_json is not valid JSON")
    item, error = _validate_model_or_error(ToolPlanItem, item)
    if error:
        return error
    return json.dumps(planning_engine.process_phase(
        phase="tool_selection", thought=thought, session_id=session_id,
        is_revision=is_revision, confidence=confidence, phase_data=item,
    ), ensure_ascii=False, indent=2)


@mcp.tool(
    name="plan_execution",
    output_schema=None,
    description="Phase 6: Define execution order. parallel_groups: semicolon-separated groups of comma-separated IDs (e.g., 'sq1,sq2;sq3').",
)
async def plan_execution(
    session_id: Annotated[str, "Session ID from plan_intent"],
    thought: Annotated[str, "Reasoning for execution order"],
    parallel_groups: Annotated[str, "Parallel batches: 'sq1,sq2;sq3,sq4' (semicolon=groups, comma=IDs)"],
    sequential: Annotated[str, "Comma-separated IDs that must run in order"],
    estimated_rounds: Annotated[int, "Estimated execution rounds"],
    confidence: Annotated[float, "Confidence 0.0-1.0"] = 1.0,
    is_revision: Annotated[bool, "True to overwrite"] = False,
) -> str:
    import json
    _, session_error = _planning_session_or_error(session_id)
    if session_error:
        return json.dumps(session_error)
    parallel = [_split_csv(g) for g in parallel_groups.split(";") if g.strip()] if parallel_groups else []
    seq = _split_csv(sequential)
    phase_data, error = _validate_model_or_error(
        ExecutionOrderOutput,
        {"parallel": parallel, "sequential": seq, "estimated_rounds": estimated_rounds},
    )
    if error:
        return error
    return json.dumps(planning_engine.process_phase(
        phase="execution_order", thought=thought, session_id=session_id,
        is_revision=is_revision, confidence=confidence,
        phase_data=phase_data,
    ), ensure_ascii=False, indent=2)


def main():
    import signal
    import os
    import threading

    # 信号处理（仅主线程）
    if threading.current_thread() is threading.main_thread():
        def handle_shutdown(signum, frame):
            os._exit(0)
        signal.signal(signal.SIGINT, handle_shutdown)
        if sys.platform != 'win32':
            signal.signal(signal.SIGTERM, handle_shutdown)

    # Windows 父进程监控
    if sys.platform == 'win32':
        import time
        import ctypes
        parent_pid = os.getppid()

        def is_parent_alive(pid):
            """Windows 下检查进程是否存活"""
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return True
            exit_code = ctypes.c_ulong()
            result = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)
            return result and exit_code.value == STILL_ACTIVE

        def monitor_parent():
            while True:
                if not is_parent_alive(parent_pid):
                    os._exit(0)
                time.sleep(2)

        threading.Thread(target=monitor_parent, daemon=True).start()

    try:
        mcp.run(transport="stdio", show_banner=False)
    except KeyboardInterrupt:
        pass
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
