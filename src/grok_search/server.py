import sys
from pathlib import Path
import json
import re
import html
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
    from grok_search.fetch_processing import augment_fetched_markdown, extract_reddit_json_post_fields
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
    from .fetch_processing import augment_fetched_markdown, extract_reddit_json_post_fields
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
_LOW_QUALITY_FETCH_MARKERS = (
    "navigation menu",
    "saved searches",
    "provide feedback",
    "github menu",
    "sign in",
    "log in",
    "请您登录后查看更多专业优质内容",
    "回到首页",
    "404 - 您访问的页面不存在",
    "可能是网址有误",
    "对应的内容已被删除",
    "处于私有状态",
    "please wait for verification",
    "js_challenge",
    "need_login=true",
    "account/unhuman",
    "找不到页面",
)
_LOW_QUALITY_UI_PREFIXES = (
    "navigation menu",
    "skip to content",
    "saved searches",
    "provide feedback",
    "sign in",
    "log in",
    "sign up",
    "create account",
    "open menu",
    "back to top",
    "open app",
    "打开app",
    "打开 app",
    "打开知乎 app",
    "打开知乎app",
    "get app",
    "use app",
    "download app",
    "登录后查看更多",
    "登录/注册后",
    "回到首页",
    "404 - 您访问的页面不存在",
    "可能是网址有误",
    "对应的内容已被删除",
    "处于私有状态",
    "please wait for verification",
    "找不到页面",
)
_SEARCH_RECOVERY_WORD_PATTERN = re.compile(r"[a-z0-9]+")
_SEARCH_RECOVERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "did",
    "do",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "my",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}
_DOMAIN_LOW_QUALITY_FETCH_MARKERS = {
    "github.com": (
        "skip to content",
        "search or jump to",
        "saved searches",
        "use saved searches to filter your results more quickly",
        "github menu",
        "there was an error while loading",
    ),
    "reddit.com": (
        "reddit - dive into anything",
        "open menu",
        "create account",
        "get app",
        "use app",
        "view in the reddit app",
        "back to top",
        "please wait for verification",
        "js_challenge",
        "solution",
    ),
    "zhihu.com": (
        "知乎，让每一次点击都充满意义",
        "打开知乎 app",
        "打开知乎app",
        "在 app 内查看完整内容",
        "登录/注册后即可查看更多内容",
        "查看全部回答",
        "写回答",
        "need_login=true",
        "account/unhuman",
        "欢迎来到知乎，发现问题背后的世界",
        "zse-ck",
    ),
    "juejin.cn": (
        "稀土掘金",
        "登录后查看更多优质内容",
        "打开app",
        "打开 app",
        "继续访问",
        "点赞",
        "评论",
        "收藏",
        "找不到页面",
        "\"statusCode\":404",
        "\"errorView\":\"NotFoundView\"",
        "verifyCenter",
    ),
    "cnblogs.com": (
        "公告",
        "昵称：",
        "园龄：",
        "粉丝：",
        "关注：",
        "积分与排名",
        "随笔档案",
        "阅读排行榜",
        "404 - 您访问的页面不存在",
        "可能是网址有误",
        "对应的内容已被删除",
        "处于私有状态",
        "邮件联系：contact@cnblogs.com",
    ),
}
_SITE_FETCH_FALLBACK_DOMAINS = {
    "github.com",
    "reddit.com",
    "zhihu.com",
    "juejin.cn",
}
_FETCH_FALLBACK_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/135.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
}
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
    async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
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


def _normalized_fetch_domain(url: str | None) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url)
    except ValueError:
        return ""
    domain = parsed.netloc.lower().split(":", 1)[0]
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _should_try_site_fetch_fallback(url: str | None) -> bool:
    domain = _normalized_fetch_domain(url)
    return any(domain == item or domain.endswith(f".{item}") for item in _SITE_FETCH_FALLBACK_DOMAINS)


def _domain_specific_fetch_markers(url: str | None) -> tuple[str, ...]:
    domain = _normalized_fetch_domain(url)
    for candidate, markers in _DOMAIN_LOW_QUALITY_FETCH_MARKERS.items():
        if domain == candidate or domain.endswith(f".{candidate}"):
            return markers
    return ()


def _extract_html_title(html_text: str) -> str:
    match = _HTML_TITLE_PATTERN.search(html_text or "")
    if not match:
        return ""
    title = html.unescape(re.sub(r"\s+", " ", match.group(1))).strip()
    return title


def _extract_meta_content(html_text: str, name: str) -> str:
    patterns = (
        rf'(?is)<meta[^>]+name=["\']{re.escape(name)}["\'][^>]+content=["\'](.*?)["\']',
        rf'(?is)<meta[^>]+content=["\'](.*?)["\'][^>]+name=["\']{re.escape(name)}["\']',
        rf'(?is)<meta[^>]+property=["\']{re.escape(name)}["\'][^>]+content=["\'](.*?)["\']',
        rf'(?is)<meta[^>]+content=["\'](.*?)["\'][^>]+property=["\']{re.escape(name)}["\']',
    )
    for pattern in patterns:
        match = re.search(pattern, html_text or "")
        if match:
            return html.unescape(re.sub(r"\s+", " ", match.group(1))).strip()
    return ""


def _build_meta_summary_markdown(
    url: str,
    title: str,
    description: str,
    source_note: str,
    *,
    fallback_mode: str = "metadata_summary",
) -> str | None:
    title = (title or "").strip()
    description = (description or "").strip()
    if not title and not description:
        return None

    parts = []
    if title:
        parts.append(f"# {title}")
    if source_note:
        parts.append(f"> {source_note}")
    parts.append("")
    if description:
        parts.append("## Summary")
        parts.append("")
        parts.append(description)
        parts.append("")
    parts.append("## Fetch Notes")
    parts.append("")
    parts.append(f"- source_url: {url}")
    parts.append(f"- fallback_mode: {fallback_mode}")
    return "\n".join(parts).strip()


def _normalize_search_recovery_result(provider: str, item: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    url = (item.get("url") or item.get("id") or "").strip()
    title = (item.get("title") or "").strip()
    content = (
        item.get("content")
        or item.get("text")
        or item.get("description")
        or item.get("snippet")
        or ""
    )
    if isinstance(content, list):
        content = " ".join(str(part).strip() for part in content if str(part).strip())
    content = str(content).strip()
    if not url and not title and not content:
        return None
    return {
        "provider": provider,
        "url": url,
        "title": title,
        "content": content,
        "score": item.get("score", 0),
    }


def _extract_search_recovery_id_tokens(url: str) -> tuple[str, ...]:
    try:
        path = urlparse(url).path or ""
    except ValueError:
        return ()
    tokens = [segment.strip() for segment in path.split("/") if segment.strip().isdigit()]
    return tuple(dict.fromkeys(tokens))


def _search_recovery_word_tokens(text: str | None) -> tuple[str, ...]:
    if not text:
        return ()
    tokens = [
        token
        for token in _SEARCH_RECOVERY_WORD_PATTERN.findall(str(text).lower())
        if len(token) >= 2 and token not in _SEARCH_RECOVERY_STOPWORDS
    ]
    return tuple(dict.fromkeys(tokens))


def _extract_reddit_recovery_signals(url: str) -> dict[str, Any]:
    signals: dict[str, Any] = {
        "domain": _normalized_fetch_domain(url),
        "path": "",
        "subreddit": "",
        "post_id": "",
        "slug": "",
        "slug_phrase": "",
        "slug_tokens": (),
    }
    try:
        path = urlparse(url).path or ""
    except ValueError:
        return signals

    normalized_path = path.rstrip("/")
    signals["path"] = normalized_path
    segments = [segment for segment in path.strip("/").split("/") if segment]
    if len(segments) >= 4 and segments[0].lower() == "r" and segments[2].lower() == "comments":
        signals["subreddit"] = segments[1]
        signals["post_id"] = segments[3].lower()
        if len(segments) >= 5:
            signals["slug"] = segments[4]
    elif len(segments) >= 2 and segments[0].lower() == "comments":
        signals["post_id"] = segments[1].lower()
        if len(segments) >= 3:
            signals["slug"] = segments[2]

    slug_phrase = re.sub(r"[-_]+", " ", str(signals["slug"] or "")).strip()
    signals["slug_phrase"] = slug_phrase
    signals["slug_tokens"] = _search_recovery_word_tokens(slug_phrase)
    return signals


def _extract_zhihu_recovery_signals(url: str) -> dict[str, Any]:
    signals: dict[str, Any] = {
        "domain": _normalized_fetch_domain(url),
        "path": "",
        "kind": "",
        "object_id": "",
        "answer_id": "",
    }
    try:
        path = urlparse(url).path or ""
    except ValueError:
        return signals

    normalized_path = path.rstrip("/")
    signals["path"] = normalized_path
    segments = [segment for segment in path.strip("/").split("/") if segment]
    domain = str(signals.get("domain") or "")
    if domain == "zhuanlan.zhihu.com" and len(segments) >= 2 and segments[0] == "p":
        signals["kind"] = "article"
        signals["object_id"] = segments[1]
    elif len(segments) >= 2 and segments[0] == "question":
        signals["kind"] = "question"
        signals["object_id"] = segments[1]
        if len(segments) >= 4 and segments[2] == "answer":
            signals["answer_id"] = segments[3]
    return signals


def _reddit_search_recovery_slug_overlap(target_url: str, item: dict[str, Any]) -> int:
    target_tokens = set(_extract_reddit_recovery_signals(target_url).get("slug_tokens") or ())
    if not target_tokens:
        return 0

    item_url = str(item.get("url") or "").strip()
    item_signals = _extract_reddit_recovery_signals(item_url) if item_url else {}
    haystack = " ".join(
        part
        for part in (
            str(item.get("title") or ""),
            str(item.get("content") or ""),
            str(item_signals.get("slug_phrase") or ""),
            item_url,
        )
        if part
    )
    item_tokens = set(_search_recovery_word_tokens(haystack))
    return len(target_tokens & item_tokens)


def _is_reliable_reddit_search_recovery_result(item: dict[str, Any], target_url: str) -> bool:
    item_url = str(item.get("url") or "").strip()
    if not item_url:
        return False

    item_domain = _normalized_fetch_domain(item_url)
    if not (item_domain == "reddit.com" or item_domain.endswith(".reddit.com")):
        return False

    target_signals = _extract_reddit_recovery_signals(target_url)
    item_signals = _extract_reddit_recovery_signals(item_url)
    target_path = str(target_signals.get("path") or "")
    item_path = str(item_signals.get("path") or "")

    if item_url == target_url or (target_path and item_path == target_path):
        return True

    combined_text = " ".join(
        part
        for part in (
            str(item.get("title") or ""),
            str(item.get("content") or ""),
            item_url,
        )
        if part
    ).lower()
    if any(
        marker in combined_text
        for marker in (
            "create account",
            "view in the reddit app",
            "open menu",
            "please wait for verification",
            "related answers",
        )
    ):
        return False

    target_post_id = str(target_signals.get("post_id") or "")
    item_post_id = str(item_signals.get("post_id") or "")
    if target_post_id:
        return bool(item_post_id and item_post_id == target_post_id)

    target_subreddit = str(target_signals.get("subreddit") or "").lower()
    item_subreddit = str(item_signals.get("subreddit") or "").lower()
    if target_subreddit and item_subreddit and target_subreddit != item_subreddit:
        return False

    return _reddit_search_recovery_slug_overlap(target_url, item) >= 2


def _is_reliable_zhihu_search_recovery_result(item: dict[str, Any], target_url: str) -> bool:
    item_url = str(item.get("url") or "").strip()
    if not item_url:
        return False

    item_domain = _normalized_fetch_domain(item_url)
    if not item_domain.endswith("zhihu.com"):
        return False

    try:
        item_path = urlparse(item_url).path or ""
    except ValueError:
        item_path = ""
    try:
        target_path = urlparse(target_url).path or ""
    except ValueError:
        target_path = ""

    title = str(item.get("title") or "").strip().lower()
    content = str(item.get("content") or "").strip().lower()
    generic_titles = {
        "知乎客户端",
        "404 - 知乎",
        "知乎大学",
        "发现- 知乎",
        "发现 - 知乎",
        "知乎，让每一次点击都充满意义",
        "知乎- 有问题，就会有答案",
        "知乎 - 有问题，就会有答案",
        "知乎专栏- 随心写作，自由表达- 知乎",
    }
    if title in {value.lower() for value in generic_titles}:
        return False

    blocked_markers = (
        "请您登录后查看更多专业优质内容",
        "立即登录/注册",
        "登录一下",
        "更多精彩内容等你发现",
        "欢迎来到知乎，发现问题背后的世界",
        "知乎，让每一次点击都充满意义",
        "请求参数异常，请升级客户端后重试",
        "need_login=true",
        "account/unhuman",
    )
    combined_text = " ".join([item_url, title, content]).lower()
    if any(marker in combined_text for marker in blocked_markers):
        return False

    if item_url == target_url or (target_path and item_path == target_path):
        return True

    return any(token and token in combined_text for token in _extract_search_recovery_id_tokens(target_url))


def _build_zhihu_search_recovery_queries(url: str) -> tuple[str, ...]:
    signals = _extract_zhihu_recovery_signals(url)
    domain = str(signals.get("domain") or "").strip()
    kind = str(signals.get("kind") or "").strip()
    object_id = str(signals.get("object_id") or "").strip()
    answer_id = str(signals.get("answer_id") or "").strip()

    queries = [f"site:{domain} {url}"]
    if kind == "question" and object_id:
        queries.append(f"site:zhihu.com/question/{object_id}")
        queries.append(f"site:zhihu.com/question {object_id}")
        if answer_id:
            queries.append(f"site:zhihu.com/question/{object_id}/answer/{answer_id}")
            queries.append(f"site:zhihu.com/answer {answer_id}")
    elif kind == "article" and object_id:
        queries.append(f"site:zhuanlan.zhihu.com/p/{object_id}")
        queries.append(f"site:zhuanlan.zhihu.com {object_id}")

    return tuple(dict.fromkeys(query.strip() for query in queries if query and query.strip()))


def _build_zhihu_identity_summary_markdown(url: str, source_note: str) -> str | None:
    domain = _normalized_fetch_domain(url)
    try:
        path = urlparse(url).path or ""
    except ValueError:
        path = ""

    segments = [segment for segment in path.strip("/").split("/") if segment]
    if domain == "zhuanlan.zhihu.com" and len(segments) >= 2 and segments[0] == "p":
        object_id = segments[1]
        title = f"知乎专栏文章 {object_id}"
        description = (
            f"该链接指向知乎专栏文章页，文章 ID 为 {object_id}。当前抓取环境命中了知乎风控或访问拦截，"
            "未恢复到可验证的公开标题或摘要，因此这里只保留可以从 URL 结构稳定确认的页面标识。\n\n"
            "如果后续提供可用登录态、浏览器态或额外外部索引，可以基于同一链接再次抓取，以恢复标题、摘要或正文。"
        )
    elif len(segments) >= 2 and segments[0] == "question":
        object_id = segments[1]
        title = f"知乎问题 {object_id}"
        description = (
            f"该链接指向知乎问题页，问题 ID 为 {object_id}。当前抓取环境命中了知乎风控或访问拦截，"
            "未恢复到可验证的公开标题或摘要，因此这里只保留可以从 URL 结构稳定确认的页面标识。\n\n"
            "如果后续提供可用登录态、浏览器态或额外外部索引，可以基于同一链接再次抓取，以恢复题目标题、摘要或正文。"
        )
    else:
        title = f"{domain} 页面"
        description = (
            "当前链接无法直接抓取正文，且没有恢复到可验证的公开标题或摘要。"
            "以下结果仅保留 URL 与站点类型等可确认信息，避免把不可靠搜索命中误判为正文。"
        )

    return _build_meta_summary_markdown(
        url,
        title,
        description,
        source_note,
        fallback_mode="url_identity_summary",
    )


def _build_reddit_identity_summary_markdown(url: str, source_note: str) -> str | None:
    signals = _extract_reddit_recovery_signals(url)
    subreddit = str(signals.get("subreddit") or "").strip()
    post_id = str(signals.get("post_id") or "").strip()
    slug_phrase = str(signals.get("slug_phrase") or "").strip()
    if not post_id:
        return None

    base_title = f"Reddit r/{subreddit} thread {post_id}" if subreddit else f"Reddit thread {post_id}"
    title = base_title
    topic_note = ""
    if slug_phrase:
        title = f'{base_title} (URL-derived topic: {slug_phrase})'
        topic_note = (
            f'URL slug suggests the topic may be "{slug_phrase}". '
            "This phrase comes from the URL structure only and is not treated as a verified public title."
        )

    description_parts = [
        (
            f"This link points to a Reddit discussion thread with post id `{post_id}`"
            + (f" under `r/{subreddit}`." if subreddit else ".")
        ),
        "The fetch path hit Reddit anti-bot protection, and search recovery did not return a reliably matched summary for the same thread.",
    ]
    if topic_note:
        description_parts.append(topic_note)
    description_parts.append(
        "The fallback keeps only stable identity signals from the URL so the result stays conservative instead of summarizing a different Reddit post."
    )
    description = "\n\n".join(description_parts)
    return _build_meta_summary_markdown(
        url,
        title,
        description,
        source_note,
        fallback_mode="url_identity_summary",
    )


def _build_reddit_search_recovery_queries(url: str) -> tuple[str, ...]:
    signals = _extract_reddit_recovery_signals(url)
    subreddit = str(signals.get("subreddit") or "").strip()
    post_id = str(signals.get("post_id") or "").strip()
    slug_phrase = str(signals.get("slug_phrase") or "").strip()
    slug_tokens = tuple(signals.get("slug_tokens") or ())

    queries = [f"site:reddit.com {url}"]
    if subreddit and post_id:
        queries.append(f"site:reddit.com/r/{subreddit} comments {post_id}")
    if subreddit and slug_phrase:
        queries.append(f'site:reddit.com/r/{subreddit} "{slug_phrase}"')
    keyword_tail = " ".join(slug_tokens[:4])
    if subreddit and (post_id or keyword_tail):
        queries.append(" ".join(part for part in ("reddit", subreddit, post_id, keyword_tail) if part).strip())

    return tuple(dict.fromkeys(query.strip() for query in queries if query and query.strip()))


def _build_reddit_json_fallback_url(url: str) -> str | None:
    signals = _extract_reddit_recovery_signals(url)
    post_id = str(signals.get("post_id") or "").strip()
    if not post_id:
        return None
    return f"https://www.reddit.com/comments/{post_id}/.json?raw_json=1"


def _build_reddit_json_fallback_markdown(
    url: str,
    text: str | None,
    source_note: str,
    *,
    recovery_provider: str = "",
) -> str | None:
    post = extract_reddit_json_post_fields(text or "")
    if not isinstance(post, dict):
        return None

    title = str(post.get("title") or "").strip()
    subreddit = str(post.get("subreddit_name_prefixed") or post.get("subreddit") or "").strip()
    author = str(post.get("author") or "").strip()
    body = str(post.get("selftext") or "").strip()
    permalink = str(post.get("permalink") or "").strip()
    score = post.get("score")
    num_comments = post.get("num_comments")
    over_18 = post.get("over_18")
    removed_by_category = str(post.get("removed_by_category") or "").strip().lower()

    title_lower = title.lower()
    if not title or title_lower in {"reddit - the heart of the internet", "reddit - dive into anything"}:
        return None
    if removed_by_category in {"deleted", "removed"}:
        return None
    if str(author).strip().lower() == "[deleted]" and body in {"", "[removed]", "[deleted]"}:
        return None
    if over_18 is True:
        return None

    target_signals = _extract_reddit_recovery_signals(url)
    target_post_id = str(target_signals.get("post_id") or "").strip().lower()
    target_subreddit = str(target_signals.get("subreddit") or "").strip().lower()
    post_id = str(post.get("id") or "").strip().lower()
    post_subreddit = subreddit.lower().removeprefix("r/") if subreddit else ""
    if target_post_id and post_id and post_id != target_post_id:
        return None
    if target_subreddit and post_subreddit and post_subreddit != target_subreddit:
        return None

    parts = [f"# {title}"]
    if source_note:
        parts.extend(["", f"> {source_note}"])
    parts.extend(["", "## Thread Metadata", ""])
    parts.append("- fallback_mode: reddit_json_summary")
    parts.append("- source_format: reddit_json")
    if recovery_provider:
        parts.append(f"- json_recovery_provider: {recovery_provider}")
    if subreddit:
        parts.append(f"- subreddit: {subreddit}")
    if author:
        parts.append(f"- author: {author}")
    if score is not None:
        parts.append(f"- score: {score}")
    if num_comments is not None:
        parts.append(f"- comment_count: {num_comments}")
    if permalink:
        parts.append(f"- permalink: https://www.reddit.com{permalink}")
    parts.append(f"- json_source_url: {_build_reddit_json_fallback_url(url) or url}")
    parts.append("")
    if body and body not in {"[removed]", "[deleted]"}:
        parts.extend(["## Post Body", "", body, ""])
    else:
        parts.extend([
            "## Post Body",
            "",
            "The Reddit JSON endpoint exposed the thread metadata, but the post body is unavailable.",
            "",
        ])
    return "\n".join(parts).strip()


def _select_best_reddit_search_recovery_result(results: list[dict[str, Any]], target_url: str) -> dict[str, Any] | None:
    reliable = [item for item in results if _is_reliable_reddit_search_recovery_result(item, target_url)]
    if not reliable:
        return None

    target_signals = _extract_reddit_recovery_signals(target_url)
    target_path = str(target_signals.get("path") or "")
    target_post_id = str(target_signals.get("post_id") or "")
    target_subreddit = str(target_signals.get("subreddit") or "").lower()

    def sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
        item_url = str(item.get("url") or "").strip()
        item_signals = _extract_reddit_recovery_signals(item_url)
        exact_url = int(bool(item_url and item_url == target_url))
        same_path = int(bool(target_path and item_signals.get("path") == target_path))
        same_post_id = int(bool(target_post_id and item_signals.get("post_id") == target_post_id))
        same_subreddit = int(
            bool(
                target_subreddit
                and str(item_signals.get("subreddit") or "").lower() == target_subreddit
            )
        )
        slug_overlap = _reddit_search_recovery_slug_overlap(target_url, item)
        content_length = len(str(item.get("content") or ""))
        title_length = len(str(item.get("title") or ""))
        score = float(item.get("score") or 0)
        return (exact_url, same_path, same_post_id, same_subreddit, slug_overlap, content_length, title_length, score)

    return max(reliable, key=sort_key)


def _select_best_search_recovery_result(results: list[dict[str, Any]], target_url: str) -> dict[str, Any] | None:
    if not results:
        return None

    target_domain = _normalized_fetch_domain(target_url)
    if target_domain == "reddit.com" or target_domain.endswith(".reddit.com"):
        return _select_best_reddit_search_recovery_result(results, target_url)

    target_path = ""
    try:
        target_path = urlparse(target_url).path or ""
    except ValueError:
        target_path = ""

    def sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
        item_url = item.get("url", "")
        item_domain = _normalized_fetch_domain(item_url)
        item_path = ""
        try:
            item_path = urlparse(item_url).path or ""
        except ValueError:
            item_path = ""

        exact_url = int(bool(item_url and item_url == target_url))
        same_path = int(bool(target_path and item_path == target_path))
        same_domain = int(bool(target_domain and item_domain == target_domain))
        content_length = len(item.get("content") or "")
        title_length = len(item.get("title") or "")
        score = float(item.get("score") or 0)
        return (exact_url, same_path, same_domain, content_length, title_length, score)

    return max(results, key=sort_key)


def _is_substantive_fetch_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if stripped.startswith(("---", "|", ">")):
        return False
    if lowered.startswith(("- ", "* ", "1. ", "2. ", "3. ", "http://", "https://", "source_url:", "inferred_title:", "fetch_structure_version:")):
        return False
    if len(stripped) >= 60:
        return True
    cjk_chars = sum(1 for ch in stripped if "\u4e00" <= ch <= "\u9fff")
    return cjk_chars >= 24


def _analyze_fetch_content(text: str | None, url: str | None = None) -> dict[str, Any]:
    content = (text or "").strip()
    lowered = content.lower()
    non_empty_lines = [line.strip() for line in content.splitlines() if line.strip()]
    preview = " ".join(non_empty_lines[:18]).lower()
    preview_lines = non_empty_lines[:18]
    marker_hits = sum(1 for marker in _LOW_QUALITY_FETCH_MARKERS if marker in lowered)
    preview_hits = sum(1 for marker in _LOW_QUALITY_FETCH_MARKERS if marker in preview)
    domain_markers = _domain_specific_fetch_markers(url)
    domain_hits = sum(1 for marker in domain_markers if marker.lower() in lowered)
    domain_preview_hits = sum(1 for marker in domain_markers if marker.lower() in preview)
    ui_line_hits = sum(
        1
        for line in preview_lines
        if any(line.lower().startswith(prefix) for prefix in _LOW_QUALITY_UI_PREFIXES)
    )
    heading_count = sum(1 for line in non_empty_lines if line.startswith("#"))
    substantive_line_count = sum(1 for line in non_empty_lines if _is_substantive_fetch_line(line))
    sentence_hits = len(re.findall(r"[。！？.!?]", content))
    code_block_count = content.count("```") // 2
    has_table = "| ---" in content or "<table" in lowered
    strong_content_signals = heading_count >= 1 or substantive_line_count >= 2 or sentence_hits >= 3 or code_block_count >= 1 or has_table

    return {
        "content": content,
        "content_length": len(content),
        "non_empty_lines": non_empty_lines,
        "preview": preview,
        "preview_lines": preview_lines,
        "marker_hits": marker_hits,
        "preview_hits": preview_hits,
        "domain_hits": domain_hits,
        "domain_preview_hits": domain_preview_hits,
        "ui_line_hits": ui_line_hits,
        "heading_count": heading_count,
        "substantive_line_count": substantive_line_count,
        "sentence_hits": sentence_hits,
        "code_block_count": code_block_count,
        "has_table": has_table,
        "strong_content_signals": strong_content_signals,
        "word_count": _word_count(content),
        "domain": _normalized_fetch_domain(url),
    }


def _is_low_quality_fetch_result(
    text: str | None,
    url: str | None = None,
    analysis: dict[str, Any] | None = None,
) -> bool:
    stats = analysis or _analyze_fetch_content(text, url)
    if not stats["content"]:
        return True

    if stats["content_length"] < 400 and stats["marker_hits"] >= 1:
        return True
    if stats["content_length"] < 1400 and stats["preview_hits"] >= 2:
        return True
    if stats["content_length"] < 2500 and stats["marker_hits"] >= 4:
        return True
    if stats["content_length"] < 500 and stats["substantive_line_count"] == 0 and not stats["strong_content_signals"]:
        return True
    if stats["ui_line_hits"] >= 4 and not stats["strong_content_signals"]:
        return True
    if (
        stats["preview_hits"] >= 2
        and stats["substantive_line_count"] <= 1
        and stats["sentence_hits"] <= 2
        and stats["content_length"] < 4000
    ):
        return True
    if stats["domain_preview_hits"] >= 2 and stats["substantive_line_count"] <= 2 and stats["heading_count"] <= 1:
        return True
    if stats["domain_hits"] >= 3 and not stats["strong_content_signals"]:
        return True
    return False


_FETCH_PROVIDER_PREFERENCE = {
    "exa": 3,
    "firecrawl": 2,
    "tavily": 1,
    "site_fallback": 1,
}

_HTML_TITLE_PATTERN = re.compile(r"(?is)<title[^>]*>(.*?)</title>")
def _build_fetch_candidate(provider: str, text: str | None, url: str) -> dict[str, Any] | None:
    analysis = _analyze_fetch_content(text, url)
    content = analysis["content"]
    if not content:
        return None

    is_low_quality = _is_low_quality_fetch_result(content, url, analysis)
    score = 0.0
    score += min(analysis["content_length"], 12000) / 260.0
    score += min(analysis["substantive_line_count"], 18) * 2.4
    score += min(analysis["heading_count"], 6) * 1.8
    score += min(analysis["sentence_hits"], 24) * 0.6
    score += min(analysis["code_block_count"], 4) * 2.0
    if analysis["has_table"]:
        score += 2.5
    score += min(analysis["word_count"], 1200) / 180.0
    score += _FETCH_PROVIDER_PREFERENCE.get(provider, 0) * 0.4
    domain = _normalized_fetch_domain(url)
    if domain == "github.com" or domain.endswith(".github.com"):
        if provider == "firecrawl":
            score -= 4.0
        elif provider in {"exa", "tavily"}:
            score += 1.0
    score -= analysis["marker_hits"] * 2.6
    score -= analysis["preview_hits"] * 1.8
    score -= analysis["domain_hits"] * 2.2
    score -= analysis["domain_preview_hits"] * 2.8
    score -= analysis["ui_line_hits"] * 3.0
    if is_low_quality:
        score -= 24.0

    return {
        "provider": provider,
        "content": content,
        "score": score,
        "is_low_quality": is_low_quality,
        "analysis": analysis,
    }


def _select_best_fetch_candidate(
    candidates: list[dict[str, Any]],
    *,
    allow_low_quality: bool = False,
) -> dict[str, Any] | None:
    if not candidates:
        return None

    pool = candidates if allow_low_quality else [candidate for candidate in candidates if not candidate["is_low_quality"]]
    if not pool:
        return None

    return max(
        pool,
        key=lambda candidate: (
            candidate["score"],
            candidate["analysis"]["content_length"],
            candidate["analysis"]["substantive_line_count"],
            _FETCH_PROVIDER_PREFERENCE.get(candidate["provider"], 0),
        ),
    )


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
        async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
            response = await client.post(endpoint, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
            if data.get("results") and len(data["results"]) > 0:
                content = data["results"][0].get("raw_content", "")
                return content if content and content.strip() else None
            return None
    except Exception as e:
        await log_info(None, f"Tavily extract error: {e}", config.debug_enabled)
        return None


async def _fetch_raw_html(url: str, ctx=None) -> str | None:
    import httpx

    try:
        async with httpx.AsyncClient(
            timeout=30.0,
            headers=_FETCH_FALLBACK_HEADERS,
            follow_redirects=True,
            trust_env=False,
        ) as client:
            response = await client.get(url)
            text = response.text
            if text and text.strip():
                return text
            response.raise_for_status()
            return None
    except Exception as e:
        await log_info(ctx, f"Raw HTML fallback error: {e}", config.debug_enabled)
        return None


async def _fetch_reddit_json_fallback(url: str, ctx=None) -> str | None:
    import httpx

    fallback_url = _build_reddit_json_fallback_url(url)
    if not fallback_url:
        return None

    headers = dict(_FETCH_FALLBACK_HEADERS)
    headers["Accept"] = "application/json,text/plain,*/*"
    try:
        async with httpx.AsyncClient(
            timeout=30.0,
            headers=headers,
            follow_redirects=True,
            trust_env=False,
        ) as client:
            response = await client.get(fallback_url)
            response.raise_for_status()
            text = response.text
            if text and text.strip():
                return text
            return None
    except Exception as e:
        await log_info(ctx, f"Reddit JSON fallback error: {e}", config.debug_enabled)
        return None


async def _recover_reddit_json_fallback_markdown(url: str, ctx=None) -> str | None:
    fallback_url = _build_reddit_json_fallback_url(url)
    if not fallback_url:
        return None

    source_note = "目标页被 Reddit 反爬拦截，回退为同帖 JSON 元数据摘要。"
    provider_attempts = [
        ("direct_json", await _fetch_reddit_json_fallback(url, ctx)),
        ("tavily_json", await _call_tavily_extract(fallback_url)),
        ("exa_json", await _call_exa_contents(fallback_url, ctx)),
        ("firecrawl_json", await _call_firecrawl_scrape(fallback_url, ctx)),
    ]
    for provider, text in provider_attempts:
        markdown = _build_reddit_json_fallback_markdown(
            url,
            text,
            source_note,
            recovery_provider=provider,
        )
        if markdown:
            return markdown
    return None


async def _build_site_fetch_fallback_candidate(url: str, ctx=None) -> dict[str, Any] | None:
    if not _should_try_site_fetch_fallback(url):
        return None

    domain = _normalized_fetch_domain(url)
    title = ""
    description = ""
    meta_markdown = None

    async def recover_via_search(note: str) -> str | None:
        candidates: list[dict[str, Any]] = []
        if domain.endswith("reddit.com"):
            queries = _build_reddit_search_recovery_queries(url)
        elif domain.endswith("zhihu.com"):
            queries = _build_zhihu_search_recovery_queries(url)
        else:
            queries = (f"site:{domain} {url}",)

        for query in queries:
            exa_results, tavily_results, firecrawl_results = await asyncio.gather(
                _call_exa_search(query, max_results=3, ctx=ctx),
                _call_tavily_search(query, max_results=3),
                _call_firecrawl_search(query, limit=3),
            )
            if exa_results:
                candidates.extend(exa_results)
            if tavily_results:
                candidates.extend(
                    item
                    for item in (
                        _normalize_search_recovery_result("tavily_search", result)
                        for result in tavily_results
                    )
                    if item
                )
            if firecrawl_results:
                candidates.extend(
                    item
                    for item in (
                        _normalize_search_recovery_result("firecrawl_search", result)
                        for result in firecrawl_results
                    )
                    if item
                )

        if domain.endswith("zhihu.com"):
            candidates = [item for item in candidates if _is_reliable_zhihu_search_recovery_result(item, url)]

        best = _select_best_search_recovery_result(candidates, url)
        if not best:
            return None
        return _build_meta_summary_markdown(
            url,
            best.get("title", "") or title,
            best.get("content", "") or description,
            note,
        )

    if domain.endswith("reddit.com"):
        reddit_json_markdown = await _recover_reddit_json_fallback_markdown(url, ctx)
        if reddit_json_markdown:
            return _build_fetch_candidate("site_fallback", reddit_json_markdown, url)

    raw_html = await _fetch_raw_html(url, ctx)
    if not raw_html:
        if domain.endswith("github.com"):
            meta_markdown = await recover_via_search("目标页抓取失败，尝试从搜索结果恢复 GitHub 页面摘要。")
            if not meta_markdown:
                return None
            return _build_fetch_candidate("site_fallback", meta_markdown, url)
        if domain.endswith("reddit.com"):
            meta_markdown = await recover_via_search("目标页抓取失败，尝试从搜索结果恢复摘要。")
            if not meta_markdown:
                meta_markdown = _build_reddit_identity_summary_markdown(
                    url,
                    "目标页抓取失败；未恢复到同帖公开摘要，回退为 URL 标识摘要。",
                )
            return _build_fetch_candidate("site_fallback", meta_markdown, url) if meta_markdown else None
        if domain.endswith("zhihu.com"):
            meta_markdown = await recover_via_search("目标页抓取失败，尝试从搜索结果恢复摘要。")
            if not meta_markdown:
                meta_markdown = _build_zhihu_identity_summary_markdown(
                    url,
                    "目标页抓取失败，回退为 URL 标识摘要。",
                )
            return _build_fetch_candidate("site_fallback", meta_markdown, url) if meta_markdown else None
        return None

    title = _extract_html_title(raw_html)
    description = (
        _extract_meta_content(raw_html, "description")
        or _extract_meta_content(raw_html, "og:description")
        or _extract_meta_content(raw_html, "twitter:description")
    )
    title = (
        title
        or _extract_meta_content(raw_html, "og:title")
        or _extract_meta_content(raw_html, "twitter:title")
    )
    lowered = raw_html.lower()

    if domain.endswith("juejin.cn"):
        if title and title != "找不到页面":
            meta_markdown = _build_meta_summary_markdown(
                url,
                title,
                description,
                "页面正文受限，回退为页面元信息摘要。",
            )
    elif domain.endswith("github.com"):
        if title or description:
            meta_markdown = _build_meta_summary_markdown(
                url,
                title or "GitHub page",
                description,
                "GitHub 页面启用站点级元信息回退。",
            )
    elif domain.endswith("reddit.com"):
        if "you've been blocked by network security" in lowered or "please wait for verification" in lowered:
            meta_markdown = await recover_via_search("目标页被反爬拦截，回退为搜索结果摘要。")
            if not meta_markdown:
                meta_markdown = _build_reddit_identity_summary_markdown(
                    url,
                    "目标页被反爬拦截；未恢复到同帖公开摘要，回退为 URL 标识摘要。",
                )
    elif domain.endswith("zhihu.com"):
        if "欢迎来到知乎，发现问题背后的世界" in raw_html or "请求存在异常" in raw_html:
            meta_markdown = await recover_via_search("目标页被知乎风控拦截，回退为搜索结果摘要。")
            if not meta_markdown:
                meta_markdown = _build_zhihu_identity_summary_markdown(
                    url,
                    "目标页被知乎风控拦截；未恢复到可验证公开摘要，回退为 URL 标识摘要。",
                )

    if not meta_markdown:
        return None

    return _build_fetch_candidate("site_fallback", meta_markdown, url)


async def _call_exa_contents(url: str, ctx=None) -> str | None:
    import httpx
    api_url = config.exa_api_url
    api_key = config.exa_api_key
    if not config.exa_enabled or not api_key:
        return None
    endpoint = f"{api_url.rstrip('/')}/contents"
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    body = {"urls": [url], "text": True}
    try:
        async with httpx.AsyncClient(timeout=90.0, trust_env=False) as client:
            response = await client.post(endpoint, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
            results = data.get("results") or []
            if not results:
                return None
            content = results[0].get("text", "")
            return content if isinstance(content, str) and content.strip() else None
    except Exception as e:
        await log_info(ctx, f"Exa error: {e}", config.debug_enabled)
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
        async with httpx.AsyncClient(timeout=90.0, trust_env=False) as client:
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


async def _call_exa_search(query: str, max_results: int = 6, ctx=None) -> list[dict] | None:
    import httpx

    api_url = config.exa_api_url
    api_key = config.exa_api_key
    if not config.exa_enabled or not api_key:
        return None

    endpoint = f"{api_url.rstrip('/')}/search"
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    body = {"query": query, "numResults": max_results}
    try:
        async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
            response = await client.post(endpoint, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
            results = data.get("results") or []
            normalized = [
                _normalize_search_recovery_result("exa_search", item)
                for item in results
            ]
            return [item for item in normalized if item]
    except Exception as e:
        await log_info(ctx, f"Exa search error: {e}", config.debug_enabled)
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
        async with httpx.AsyncClient(timeout=90.0, trust_env=False) as client:
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
    domain = _normalized_fetch_domain(url)
    if domain == "github.com" or domain.endswith(".github.com"):
        await log_info(ctx, "Firecrawl skipped for GitHub URL; prefer Tavily/Exa/raw_html.", config.debug_enabled)
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
            async with httpx.AsyncClient(timeout=90.0, trust_env=False) as client:
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

    candidates: list[dict[str, Any]] = []

    tavily_result, firecrawl_result, exa_result = await asyncio.gather(
        _call_tavily_extract(url),
        _call_firecrawl_scrape(url, ctx),
        _call_exa_contents(url, ctx),
    )

    tavily_candidate = _build_fetch_candidate("tavily", tavily_result, url)
    if tavily_candidate:
        candidates.append(tavily_candidate)
        await log_info(
            ctx,
            f"Tavily candidate collected: score={tavily_candidate['score']:.1f}, low_quality={tavily_candidate['is_low_quality']}",
            config.debug_enabled,
        )
    else:
        await log_info(ctx, "Tavily unavailable or failed.", config.debug_enabled)

    firecrawl_candidate = _build_fetch_candidate("firecrawl", firecrawl_result, url)
    if firecrawl_candidate:
        candidates.append(firecrawl_candidate)
        await log_info(
            ctx,
            f"Firecrawl candidate collected: score={firecrawl_candidate['score']:.1f}, low_quality={firecrawl_candidate['is_low_quality']}",
            config.debug_enabled,
        )
    else:
        await log_info(ctx, "Firecrawl unavailable or failed.", config.debug_enabled)

    exa_candidate = _build_fetch_candidate("exa", exa_result, url)
    if exa_candidate:
        candidates.append(exa_candidate)
        await log_info(
            ctx,
            f"Exa candidate collected: score={exa_candidate['score']:.1f}, low_quality={exa_candidate['is_low_quality']}",
            config.debug_enabled,
        )
    else:
        await log_info(ctx, "Exa unavailable or failed.", config.debug_enabled)

    best_candidate = _select_best_fetch_candidate(candidates)
    if not best_candidate and _should_try_site_fetch_fallback(url):
        fallback_candidate = await _build_site_fetch_fallback_candidate(url, ctx)
        if fallback_candidate:
            candidates.append(fallback_candidate)
            best_candidate = _select_best_fetch_candidate(candidates)
            await log_info(
                ctx,
                f"Site fallback candidate collected: score={fallback_candidate['score']:.1f}, low_quality={fallback_candidate['is_low_quality']}",
                config.debug_enabled,
            )

    if best_candidate:
        await log_info(
            ctx,
            f"Fetch Finished ({best_candidate['provider'].title()})! selected score={best_candidate['score']:.1f}",
            config.debug_enabled,
        )
        return augment_fetched_markdown(url, best_candidate["content"])

    if candidates:
        best_low_quality = _select_best_fetch_candidate(candidates, allow_low_quality=True)
        if best_low_quality:
            await log_info(
                ctx,
                f"All candidates were low-quality; best provider={best_low_quality['provider']} score={best_low_quality['score']:.1f}",
                config.debug_enabled,
            )

    await log_info(ctx, "Fetch Failed!", config.debug_enabled)
    if not config.tavily_api_key and not config.firecrawl_api_key and not config.exa_api_key:
        return "配置错误: TAVILY_API_KEY、FIRECRAWL_API_KEY 和 EXA_API_KEY 均未配置"
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
        async with httpx.AsyncClient(timeout=float(timeout + 10), trust_env=False) as client:
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

        async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
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
