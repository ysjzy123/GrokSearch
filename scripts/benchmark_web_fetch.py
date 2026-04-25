#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tomllib
from pathlib import Path
from time import perf_counter
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from grok_search import server  # noqa: E402


DEFAULT_PROVIDERS = ("tavily", "firecrawl", "exa")
DEFAULT_CODEX_CONFIG_PATH = Path.home() / ".codex" / "config.toml"
LIVE_ENV_KEYS = [
    "GROK_API_URL",
    "GROK_API_KEY",
    "TAVILY_API_URL",
    "TAVILY_API_KEY",
    "TAVILY_ENABLED",
    "FIRECRAWL_API_URL",
    "FIRECRAWL_API_KEY",
    "EXA_API_URL",
    "EXA_API_KEY",
    "EXA_ENABLED",
    "GUDA_API_KEY",
    "GUDA_BASE_URL",
    "GROK_MODEL",
]
PROXY_ENV_KEYS = [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "all_proxy",
    "no_proxy",
    "NO_PROXY",
]
FINAL_FETCH_METADATA_KEYS = (
    "source_url",
    "inferred_title",
    "fetch_structure_version",
    "fallback_mode",
    "source_format",
    "json_recovery_provider",
)


def _stringify_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _load_live_env_from_codex_config(config_path: Path | None = None) -> dict[str, str]:
    target = config_path or DEFAULT_CODEX_CONFIG_PATH
    if not target.exists():
        return {}

    try:
        data = tomllib.loads(target.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError):
        return {}

    server_env = data.get("mcp_servers", {}).get("grok-search", {}).get("env", {})
    if not isinstance(server_env, dict):
        return {}

    loaded: dict[str, str] = {}
    for key in LIVE_ENV_KEYS:
        if key in server_env and server_env[key] is not None:
            loaded[key] = _stringify_env_value(server_env[key])
    return loaded


def _apply_live_env_from_codex_config(
    config_path: Path | None = None,
    *,
    prefer_existing: bool = True,
) -> tuple[Path, list[str]]:
    target = config_path or DEFAULT_CODEX_CONFIG_PATH
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)

    loaded = _load_live_env_from_codex_config(target)
    if prefer_existing:
        for key, value in loaded.items():
            os.environ.setdefault(key, value)
    else:
        os.environ.update(loaded)

    server.config._cached_model = None
    return target, sorted(loaded)


def _load_urls(url_args: list[str], url_file: str | None) -> list[str]:
    urls: list[str] = []
    if url_file:
        for line in Path(url_file).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                urls.append(stripped)
    urls.extend(url_args)

    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def _parse_providers(raw: str) -> list[str]:
    items = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = [item for item in items if item not in DEFAULT_PROVIDERS]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported providers: {', '.join(invalid)}")
    if not items:
        raise argparse.ArgumentTypeError("At least one provider is required")
    return items


def _provider_state(provider: str) -> dict[str, Any]:
    if provider == "tavily":
        return {
            "enabled": bool(server.config.tavily_enabled),
            "configured": bool(server.config.tavily_enabled and server.config.tavily_api_key),
            "reason": "disabled" if not server.config.tavily_enabled else ("missing_key" if not server.config.tavily_api_key else "ready"),
        }
    if provider == "firecrawl":
        return {
            "enabled": True,
            "configured": bool(server.config.firecrawl_api_key),
            "reason": "missing_key" if not server.config.firecrawl_api_key else "ready",
        }
    if provider == "exa":
        return {
            "enabled": bool(server.config.exa_enabled),
            "configured": bool(server.config.exa_enabled and server.config.exa_api_key),
            "reason": "disabled" if not server.config.exa_enabled else ("missing_key" if not server.config.exa_api_key else "ready"),
        }
    raise ValueError(f"Unknown provider: {provider}")


async def _fetch_provider(provider: str, url: str) -> dict[str, Any]:
    state = _provider_state(provider)
    if not state["configured"]:
        return {
            "provider": provider,
            "status": state["reason"],
            "elapsed_ms": 0.0,
            "candidate": None,
        }

    started = perf_counter()
    if provider == "tavily":
        text = await server._call_tavily_extract(url)
    elif provider == "firecrawl":
        text = await server._call_firecrawl_scrape(url, None)
    elif provider == "exa":
        text = await server._call_exa_contents(url, None)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    elapsed_ms = (perf_counter() - started) * 1000.0

    candidate = server._build_fetch_candidate(provider, text, url)
    if candidate is None:
        return {
            "provider": provider,
            "status": "empty_or_failed",
            "elapsed_ms": elapsed_ms,
            "candidate": None,
        }
    return {
        "provider": provider,
        "status": "ok",
        "elapsed_ms": elapsed_ms,
        "candidate": candidate,
    }


def _candidate_preview(candidate: dict[str, Any] | None, limit: int) -> str:
    if not candidate:
        return ""
    content = candidate["content"].strip().replace("\r\n", "\n")
    preview = " ".join(content.splitlines()[:6]).strip()
    if len(preview) <= limit:
        return preview
    return preview[: max(0, limit - 3)] + "..."


def _serialize_attempt(attempt: dict[str, Any], preview_chars: int) -> dict[str, Any]:
    candidate = attempt["candidate"]
    if candidate is None:
        return {
            "provider": attempt["provider"],
            "status": attempt["status"],
            "elapsed_ms": round(attempt["elapsed_ms"], 1),
        }

    analysis = candidate["analysis"]
    return {
        "provider": attempt["provider"],
        "status": attempt["status"],
        "elapsed_ms": round(attempt["elapsed_ms"], 1),
        "score": round(candidate["score"], 2),
        "is_low_quality": candidate["is_low_quality"],
        "content_length": analysis["content_length"],
        "word_count": analysis["word_count"],
        "heading_count": analysis["heading_count"],
        "substantive_line_count": analysis["substantive_line_count"],
        "sentence_hits": analysis["sentence_hits"],
        "code_block_count": analysis["code_block_count"],
        "has_table": analysis["has_table"],
        "marker_hits": analysis["marker_hits"],
        "preview_hits": analysis["preview_hits"],
        "domain_hits": analysis["domain_hits"],
        "domain_preview_hits": analysis["domain_preview_hits"],
        "ui_line_hits": analysis["ui_line_hits"],
        "preview": _candidate_preview(candidate, preview_chars),
    }


def _slugify_url(url: str) -> str:
    cleaned = []
    for ch in url:
        if ch.isalnum():
            cleaned.append(ch.lower())
        else:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug[:120] or "url"


def _save_candidate_artifacts(output_dir: Path, url: str, attempts: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = _slugify_url(url)
    for attempt in attempts:
        candidate = attempt["candidate"]
        if candidate is None:
            continue
        target = output_dir / f"{base}.{attempt['provider']}.md"
        target.write_text(candidate["content"], encoding="utf-8")


def _coerce_metadata_value(key: str, value: str) -> Any:
    if key == "fetch_structure_version" and re.fullmatch(r"\d+", value):
        return int(value)
    return value


def _extract_final_fetch_metadata(text: str | None) -> dict[str, Any]:
    if not isinstance(text, str):
        return {}

    stripped = text.strip()
    if not stripped or stripped.startswith(("配置错误:", "提取失败:", "无效URL:")):
        return {}

    metadata: dict[str, Any] = {}
    for raw_line in stripped.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        normalized = line[2:].strip() if line.startswith("- ") else line
        for key in FINAL_FETCH_METADATA_KEYS:
            prefix = f"{key}:"
            if normalized.startswith(prefix):
                value = normalized[len(prefix):].strip()
                if value and key not in metadata:
                    metadata[key] = _coerce_metadata_value(key, value)
                break
    return metadata


async def _benchmark_one_url(
    url: str,
    providers: list[str],
    preview_chars: int,
    artifact_dir: Path | None,
) -> dict[str, Any]:
    attempts = await asyncio.gather(*[_fetch_provider(provider, url) for provider in providers])
    candidates = [attempt["candidate"] for attempt in attempts if attempt["candidate"] is not None]

    winner = server._select_best_fetch_candidate(candidates)
    best_low_quality = server._select_best_fetch_candidate(candidates, allow_low_quality=True)
    final_fetch = await server.web_fetch(url)
    final_fetch_ok = isinstance(final_fetch, str) and not final_fetch.startswith(("配置错误:", "提取失败:", "无效URL:"))
    final_fetch_metadata = _extract_final_fetch_metadata(final_fetch if final_fetch_ok else None)

    if artifact_dir is not None:
        _save_candidate_artifacts(artifact_dir, url, attempts)

    return {
        "url": url,
        "winner": winner["provider"] if winner else None,
        "winner_score": round(winner["score"], 2) if winner else None,
        "best_low_quality_provider": best_low_quality["provider"] if best_low_quality and not winner else None,
        "final_fetch_ok": final_fetch_ok,
        "final_fetch_length": len(final_fetch) if final_fetch_ok else 0,
        "final_fetch_preview": _candidate_preview({"content": final_fetch}, preview_chars) if final_fetch_ok else (final_fetch or ""),
        "final_fetch_metadata": final_fetch_metadata,
        "providers": [_serialize_attempt(attempt, preview_chars) for attempt in attempts],
    }


def _build_summary(results: list[dict[str, Any]], providers: list[str]) -> dict[str, Any]:
    checked_urls = [item["url"] for item in results]
    winner_counts: dict[str, int] = {}
    successful_urls = 0
    final_fetch_successful_urls = 0
    final_fetch_structured_urls = 0
    final_fetch_fallback_urls = 0
    all_low_quality_urls = 0
    final_fetch_fallback_mode_counts: dict[str, int] = {}
    final_fetch_source_format_counts: dict[str, int] = {}
    final_fetch_recovery_provider_counts: dict[str, int] = {}
    provider_stats: dict[str, dict[str, Any]] = {
        provider: {
            "attempted_urls": 0,
            "successful_urls": 0,
            "non_low_quality_urls": 0,
            "total_length": 0,
            "total_elapsed_ms": 0.0,
            "winner_count": 0,
        }
        for provider in providers
    }
    summaries: list[dict[str, Any]] = []

    for item in results:
        if item["winner"]:
            successful_urls += 1
            winner_counts[item["winner"]] = winner_counts.get(item["winner"], 0) + 1
            if item["winner"] in provider_stats:
                provider_stats[item["winner"]]["winner_count"] += 1
        if item["final_fetch_ok"]:
            final_fetch_successful_urls += 1
        final_fetch_metadata = item.get("final_fetch_metadata") or {}
        if final_fetch_metadata:
            final_fetch_structured_urls += 1
        fallback_mode = str(final_fetch_metadata.get("fallback_mode") or "").strip()
        if fallback_mode:
            final_fetch_fallback_urls += 1
            final_fetch_fallback_mode_counts[fallback_mode] = final_fetch_fallback_mode_counts.get(fallback_mode, 0) + 1
        source_format = str(final_fetch_metadata.get("source_format") or "").strip()
        if source_format:
            final_fetch_source_format_counts[source_format] = final_fetch_source_format_counts.get(source_format, 0) + 1
        recovery_provider = str(final_fetch_metadata.get("json_recovery_provider") or "").strip()
        if recovery_provider:
            final_fetch_recovery_provider_counts[recovery_provider] = final_fetch_recovery_provider_counts.get(recovery_provider, 0) + 1

        url_has_non_low_quality = False
        attempts_summary: list[dict[str, Any]] = []
        attempts_by_provider = {provider_info["provider"]: provider_info for provider_info in item["providers"]}
        for provider in providers:
            provider_info = attempts_by_provider.get(provider)
            if provider_info is None:
                continue

            stats = provider_stats[provider]
            stats["attempted_urls"] += 1
            stats["total_elapsed_ms"] += float(provider_info.get("elapsed_ms", 0.0))

            content_length = int(provider_info.get("content_length", 0) or 0)
            if provider_info.get("status") == "ok" and content_length > 0:
                stats["successful_urls"] += 1
                stats["total_length"] += content_length

            is_low_quality = provider_info.get("is_low_quality")
            if provider_info.get("status") == "ok" and is_low_quality is False:
                stats["non_low_quality_urls"] += 1
                url_has_non_low_quality = True

            attempts_summary.append(
                {
                    "provider": provider,
                    "status": provider_info.get("status"),
                    "elapsed_s": round(float(provider_info.get("elapsed_ms", 0.0)) / 1000.0, 2),
                    "content_length": content_length,
                    "low_quality": is_low_quality,
                    "score": provider_info.get("score"),
                }
            )

        if not url_has_non_low_quality:
            all_low_quality_urls += 1

        summaries.append(
            {
                "url": item["url"],
                "winner": item["winner"],
                "winner_score": item.get("winner_score"),
                "best_low_quality_provider": item.get("best_low_quality_provider"),
                "final_fetch_ok": item["final_fetch_ok"],
                "final_fetch_length": item["final_fetch_length"],
                "final_fetch_metadata": final_fetch_metadata,
                "attempts": attempts_summary,
            }
        )

    provider_summary: dict[str, dict[str, Any]] = {}
    for provider in providers:
        stats = provider_stats[provider]
        attempted_urls = stats["attempted_urls"]
        success_denominator = max(1, attempted_urls)
        successful_urls_for_provider = stats["successful_urls"]
        provider_summary[provider] = {
            "attempted_urls": attempted_urls,
            "successful_urls": successful_urls_for_provider,
            "success_rate": round(successful_urls_for_provider / success_denominator, 3),
            "non_low_quality_urls": stats["non_low_quality_urls"],
            "non_low_quality_rate": round(stats["non_low_quality_urls"] / success_denominator, 3),
            "winner_count": stats["winner_count"],
            "avg_content_length": round(stats["total_length"] / max(1, successful_urls_for_provider), 1),
            "avg_elapsed_s": round((stats["total_elapsed_ms"] / 1000.0) / success_denominator, 2),
        }

    total_urls = max(1, len(results))
    return {
        "checked_urls": checked_urls,
        "checked_url_count": len(results),
        "provider_order": list(providers),
        "successful_urls": successful_urls,
        "success_rate": round(successful_urls / total_urls, 3),
        "final_fetch_successful_urls": final_fetch_successful_urls,
        "final_fetch_success_rate": round(final_fetch_successful_urls / total_urls, 3),
        "final_fetch_structured_urls": final_fetch_structured_urls,
        "final_fetch_structured_rate": round(final_fetch_structured_urls / total_urls, 3),
        "final_fetch_fallback_urls": final_fetch_fallback_urls,
        "final_fetch_fallback_rate": round(final_fetch_fallback_urls / total_urls, 3),
        "final_fetch_fallback_mode_counts": final_fetch_fallback_mode_counts,
        "final_fetch_source_format_counts": final_fetch_source_format_counts,
        "final_fetch_recovery_provider_counts": final_fetch_recovery_provider_counts,
        "all_low_quality_urls": all_low_quality_urls,
        "all_low_quality_rate": round(all_low_quality_urls / total_urls, 3),
        "winner_counts": winner_counts,
        "provider_summary": provider_summary,
        "summaries": summaries,
    }


async def _run_benchmark(
    urls: list[str],
    providers: list[str],
    preview_chars: int,
    concurrency: int,
    artifact_dir: Path | None,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: list[dict[str, Any]] = []

    async def runner(url: str) -> None:
        async with semaphore:
            results.append(await _benchmark_one_url(url, providers, preview_chars, artifact_dir))

    await asyncio.gather(*(runner(url) for url in urls))
    order = {url: index for index, url in enumerate(urls)}
    results.sort(key=lambda item: order[item["url"]])
    return results


def _print_report(results: list[dict[str, Any]], providers: list[str], summary: dict[str, Any] | None = None) -> None:
    if summary is None:
        summary = _build_summary(results, providers)

    print("== Fetch Benchmark ==")
    print(f"providers: {', '.join(providers)}")
    for provider in providers:
        state = _provider_state(provider)
        print(
            f"- {provider}: enabled={str(state['enabled']).lower()} "
            f"configured={str(state['configured']).lower()} state={state['reason']}"
        )

    print("")
    for item in results:
        print(f"URL: {item['url']}")
        print(
            f"winner: {item['winner'] or 'none'}"
            + (f" (score={item['winner_score']})" if item["winner_score"] is not None else "")
        )
        print(
            f"final_fetch: {'ok' if item['final_fetch_ok'] else 'failed'}"
            + (f", len={item['final_fetch_length']}" if item["final_fetch_ok"] else "")
        )
        final_fetch_metadata = item.get("final_fetch_metadata") or {}
        if final_fetch_metadata:
            meta_parts = []
            if final_fetch_metadata.get("fallback_mode"):
                meta_parts.append(f"mode={final_fetch_metadata['fallback_mode']}")
            if final_fetch_metadata.get("json_recovery_provider"):
                meta_parts.append(f"recovery={final_fetch_metadata['json_recovery_provider']}")
            if final_fetch_metadata.get("source_format"):
                meta_parts.append(f"format={final_fetch_metadata['source_format']}")
            if meta_parts:
                print(f"final_fetch_meta: {', '.join(meta_parts)}")
        if item.get("final_fetch_preview"):
            print(f"final_preview: {item['final_fetch_preview']}")
        if item["best_low_quality_provider"]:
            print(f"best_low_quality_only: {item['best_low_quality_provider']}")
        for provider_info in item["providers"]:
            line = (
                f"  - {provider_info['provider']}: status={provider_info['status']}"
                f", elapsed_ms={provider_info['elapsed_ms']}"
            )
            if provider_info["status"] == "ok":
                line += (
                    f", score={provider_info['score']}"
                    f", low_quality={str(provider_info['is_low_quality']).lower()}"
                    f", len={provider_info['content_length']}"
                    f", headings={provider_info['heading_count']}"
                    f", substantive={provider_info['substantive_line_count']}"
                    f", markers={provider_info['marker_hits']}/{provider_info['domain_hits']}"
                )
            print(line)
            preview = provider_info.get("preview")
            if preview:
                print(f"    preview: {preview}")
        print("")

    print("== Summary ==")
    success_count = summary["successful_urls"]
    final_fetch_success_count = summary["final_fetch_successful_urls"]
    print(f"urls={summary['checked_url_count']} success={success_count} failed={len(results) - success_count}")
    print(
        f"final_fetch_success={final_fetch_success_count} "
        f"final_fetch_failed={len(results) - final_fetch_success_count}"
    )
    if summary["final_fetch_fallback_mode_counts"]:
        fallback_modes = ", ".join(
            f"{mode}={count}" for mode, count in sorted(summary["final_fetch_fallback_mode_counts"].items())
        )
        print(f"final_fetch_fallback_modes: {fallback_modes}")
    if summary["final_fetch_recovery_provider_counts"]:
        recovery_providers = ", ".join(
            f"{provider}={count}" for provider, count in sorted(summary["final_fetch_recovery_provider_counts"].items())
        )
        print(f"final_fetch_recovery_providers: {recovery_providers}")
    if summary["winner_counts"]:
        winners = ", ".join(f"{provider}={count}" for provider, count in sorted(summary["winner_counts"].items()))
        print(f"winner_counts: {winners}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark raw web_fetch candidates from Tavily, Firecrawl, and Exa on real URLs.",
    )
    parser.add_argument("--url", action="append", default=[], help="Benchmark a URL. Repeat for multiple URLs.")
    parser.add_argument("--url-file", help="Read benchmark URLs from a UTF-8 text file, one URL per line.")
    parser.add_argument(
        "--providers",
        default="tavily,firecrawl,exa",
        help="Comma-separated provider list. Default: tavily,firecrawl,exa",
    )
    parser.add_argument("--preview-chars", type=int, default=180, help="Max preview characters per provider.")
    parser.add_argument("--concurrency", type=int, default=2, help="Number of URLs to benchmark concurrently.")
    parser.add_argument("--json-out", help="Optional path to write full JSON results.")
    parser.add_argument(
        "--codex-config",
        help=f"Optional Codex config.toml path. Default: {DEFAULT_CODEX_CONFIG_PATH}",
    )
    parser.add_argument(
        "--no-codex-env",
        action="store_true",
        help="Do not auto-load MCP env from Codex config.toml before running the benchmark.",
    )
    parser.add_argument(
        "--codex-env-override",
        action="store_true",
        help="Override current environment values with Codex config values when both are present.",
    )
    parser.add_argument(
        "--artifact-dir",
        help="Optional directory to store raw provider markdown outputs for each URL.",
    )
    return parser


async def _async_main(args: argparse.Namespace) -> int:
    if not args.no_codex_env:
        codex_config = Path(args.codex_config).expanduser() if args.codex_config else None
        config_path, loaded_keys = _apply_live_env_from_codex_config(
            codex_config,
            prefer_existing=not args.codex_env_override,
        )
        if loaded_keys:
            print(f"Loaded Codex MCP env from {config_path}: {', '.join(loaded_keys)}")
        else:
            print(f"No Codex MCP env loaded from {config_path}; using current environment.")

    providers = _parse_providers(args.providers)
    urls = _load_urls(args.url, args.url_file)
    if not urls:
        print("No URLs provided. Use --url or --url-file.", file=sys.stderr)
        return 2

    invalid_urls = [url for url in urls if not server._is_valid_web_url(url)]
    if invalid_urls:
        print(f"Invalid URLs: {', '.join(invalid_urls)}", file=sys.stderr)
        return 2

    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else None
    results = await _run_benchmark(
        urls=urls,
        providers=providers,
        preview_chars=max(40, args.preview_chars),
        concurrency=max(1, args.concurrency),
        artifact_dir=artifact_dir,
    )
    summary = _build_summary(results, providers)
    _print_report(results, providers, summary)

    if args.json_out:
        payload = {
            "providers": providers,
            "summary": summary,
            "results": results,
        }
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON written to: {args.json_out}")
    return 0


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
