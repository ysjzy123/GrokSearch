# Grok Search Improvement Workflow

## Goal

This project should evolve only when a candidate version demonstrates broad improvement over the original GitHub baseline instead of fixing one issue while regressing others.

The baseline reference is:

- `origin/grok-with-tavily`

The local candidate is the editable working tree at:

- `/home/rk/grok-search-local`

The score runner must be executed from the project environment:

- `uv run python scripts/compare_versions.py`

This is intentional. The project depends on `fastmcp`, and the comparator also needs to isolate the target repo from the editable install that powers daily development.

## Design Principles

The workflow is strict on purpose:

1. Compare against the original GitHub baseline every time.
2. Prefer deterministic probes before live-network probes.
3. Treat guardrail regressions as hard failures, not tradeoffs.
4. Keep each implementation batch small enough that score deltas remain explainable.
5. Record the decision trail outside the repo in Obsidian so engineering context survives local shell history.

## Change Policy

Every improvement follows the same loop:

1. Define the target problem and intended behavior.
2. Record the design rationale before implementation.
3. Implement the smallest coherent change set.
4. Run deterministic score comparison against the baseline.
5. Run optional live smoke checks when network access is available.
6. Keep the change only when:
   - total score is higher than baseline
   - every category score is greater than or equal to baseline
   - no new gating probe fails relative to baseline
7. Append the outcome to the Obsidian engineering note.

## Baseline Isolation

The comparator runs each repo in a separate subprocess with:

- `python -S` to skip editable-install `.pth` hooks
- an explicit `PYTHONPATH` containing the target repo `src/` and only required site-packages
- a repo-local working directory

Without this isolation, the baseline can accidentally import the current editable working tree and produce false scores.

## Scoring Categories

The scorecard is implemented by `scripts/compare_versions.py`.

Deterministic categories:

- `contract` (10): provider interface consistency and exported API sanity.
- `validation` (20): input validation and prevention of silent bad states.
- `source_quality` (25): source extraction, default extra sources, enrichment, and ranking.
- `query_quality` (15): conditional time-context injection and prompt hygiene.
- `resilience` (15): surfaced errors instead of silent failure and balanced source budgeting.
- `cache_quality` (10): persisted source-cache recovery and pruning behavior.
- `search_coverage` (10): broad-query expansion and multi-facet recall.
- `aggregation_breadth` (10): multi-engine, multi-query aggregation quality.
- `evidence_binding` (10): answer-claim linkage to concrete supporting sources.
- `citation_precision` (10): claim bindings prioritize specific evidence and suppress generic decoys.
- `efficiency` (20): budget-aware fan-out, early stop, and follow-up query efficiency.
- `engineering` (15): regression tests and repository hygiene for local development artifacts.

Optional live categories:

- `live` (20): real network smoke checks for no-think leakage, source availability, and basic answer quality.
- `retrieval_robustness` (20): difficult-site fetch/map checks against real pages such as GitHub and Linux.do.
- `live_efficiency` (20): real latency and budget-usage checks under live network conditions.

## Probe Matrix

Each category contains a mix of hard-gating probes and soft quality probes.

- `contract`
  - Gating: provider `search()` returns `str`
  - Gating: base provider defines `fetch()`
- `validation`
  - Gating: invalid complexity level is rejected
  - Gating: first search term requires `approach`
  - Gating: invalid fetch URL is rejected
  - Gating: invalid model is not persisted
  - Gating: `toggle_builtin_tools` writes to a local workspace path outside git repos
- `source_quality`
  - Gating: `<think>` blocks are stripped
  - Gating: default `extra_sources` produces extra references
  - Gating: Tavily and Firecrawl budgets are balanced when both exist
  - Quality: sparse source metadata is enriched
  - Quality: sources are ranked after enrichment
- `query_quality`
  - Gating: timeless queries skip time-context injection
  - Gating: time-sensitive queries include time-context injection
- `resilience`
  - Gating: provider exceptions surface as explicit user-facing errors
  - Gating: extra-source budget uses both backends when available
- `cache_quality`
  - Gating: source cache survives process restart through persisted storage
  - Gating: persisted cache prunes old sessions when over capacity
- `search_coverage`
  - Gating: broad queries expand into multiple planned searches
  - Gating: expanded searches capture multiple useful facets
- `aggregation_breadth`
  - Gating: search trace records multi-engine aggregation
  - Gating: aggregated sources preserve query-level provenance
- `evidence_binding`
  - Gating: answer claims receive structured source bindings
  - Gating: bindings match claim-specific sources
- `citation_precision`
  - Gating: claim bindings prioritize the most specific top source
  - Gating: generic decoy sources are filtered from claim bindings
- `efficiency`
  - Gating: broad queries respect a bounded external fan-out when early support is already strong
  - Gating: unused budget is preserved when early stop triggers
  - Gating: follow-up expansion executes only when initial coverage is weak
  - Gating: extra fan-out must deliver additional sources instead of pure overhead
- `engineering`
  - Gating: regression tests exist
  - Quality: local development artifacts are ignored
- `live`
  - Gating: no `<think>` leakage in timeless query
  - Gating: no `<think>` leakage in time-sensitive query
  - Gating: non-empty sources returned for basic factual lookup
- `retrieval_robustness`
  - Quality: GitHub repository fetch returns substantial content
  - Quality: Linux.do topic fetch returns substantial content
  - Quality: site mapping returns useful link structure when supported
- `live_efficiency`
  - Quality: easy live queries use less than the maximum extra-source budget
  - Quality: easy live queries remain within a reasonable latency budget
  - Quality: hard live queries spend follow-up budget only when it buys additional coverage
  - Quality: hard live queries remain within a bounded latency budget

## Acceptance Gate

A candidate is accepted only if:

- `candidate.total > baseline.total`
- for every category: `candidate.category >= baseline.category`
- `new_failed_gating_probes` is empty

Historical baseline defects may still appear in `remaining_failed_gating_probes`. They should be logged and prioritized, but they do not automatically block a change unless the candidate introduces a new gating failure or regresses an existing category.

## Execution Sequence

Use this exact sequence for every change batch:

1. Write or update the design note for the target change.
2. Implement the smallest coherent patch.
3. Run `uv run --with pytest --with pytest-asyncio pytest -q`.
4. Run `uv run python scripts/compare_versions.py`.
5. Run `uv run python scripts/compare_versions.py --include-live` when network is available.
6. When the change touches `web_fetch`, also run `uv run python scripts/compare_versions.py --include-live --benchmark-file benchmarks/web_fetch_real_urls.txt`.
7. Optionally run `uv run python scripts/benchmark_web_fetch.py --url-file benchmarks/web_fetch_real_urls.txt --json-out benchmarks/live_fetch_benchmark_result.json` for per-provider inspection.
8. Compare category deltas and gating failures.
9. Append the outcome to Obsidian with keep/revert decision.

## Obsidian Logging Protocol

Every landed change appends a dated section containing:

- change summary
- files touched
- deterministic score delta
- live smoke results when available
- keep/revert decision

Recommended note path:

- `/mnt/f/Obsidian Vault/自建服务/grok-search 评分标准与聚合搜索分析 2026-04-09.md`

Suggested section template:

```md
## YYYY-MM-DD HH:MM

- change: concise description
- files: comma-separated touched files
- deterministic: `candidate X / baseline Y`, include category deltas
- live: pass/fail summary or "not run"
- decision: keep or revert
- follow-up: next risk or next improvement
```

## Live Fetch Benchmark Notes

Use the benchmark sample file at:

- `benchmarks/web_fetch_real_urls.txt`

Use the benchmark runner when you need provider-level visibility instead of only score deltas:

- `uv run python scripts/benchmark_web_fetch.py --url-file benchmarks/web_fetch_real_urls.txt --json-out benchmarks/live_fetch_benchmark_result.json`
- add `--artifact-dir benchmarks/live_fetch_artifacts` only when you need raw provider markdown outputs for manual review

Treat the following as generated artifacts and keep them out of git unless there is a deliberate reason to preserve them:

- `benchmarks/live_fetch_artifacts*/`
- `benchmarks/live_*.json`
- ad-hoc `*.bak.*` backups

## Current Improvement Themes

The current roadmap prioritizes:

1. query quality and prompt hygiene
2. source quality and ranking
3. search coverage and aggregation depth
4. evidence binding for answer claims
5. citation precision and claim-level evidence quality
6. budget-aware efficiency and early-stop behavior
7. cache persistence
8. live benchmark expansion
9. maintainability and provider extensibility
