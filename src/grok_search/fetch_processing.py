from __future__ import annotations

import html
import json
import re
from urllib.parse import urlparse


_ATX_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_YAML_FIELD_PATTERN = re.compile(r"^[A-Za-z0-9_-]+:\s+.+$")
_MARKDOWN_TABLE_SEPARATOR_PATTERN = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")
_HTML_TABLE_PATTERN = re.compile(r"(?is)<table\b[^>]*>.*?</table>")
_HTML_ROW_BLOCK_PATTERN = re.compile(r"(?is)(?:<tr\b[^>]*>.*?</tr>\s*){2,}")
_HTML_ROW_PATTERN = re.compile(r"(?is)<tr\b[^>]*>(.*?)</tr>")
_HTML_CELL_PATTERN = re.compile(r"(?is)<t(?P<tag>[hd])\b[^>]*>(.*?)</t[hd]>")
_HTML_TAG_PATTERN = re.compile(r"(?is)<[^>]+>")

_MAX_HEADING_ITEMS = 12
_MAX_TABLE_COUNT = 4
_MAX_TABLE_ROWS = 8


def _clean_markdown_text(value: str) -> str:
    cleaned = value.strip()
    cleaned = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", cleaned)
    cleaned = re.sub(r"[`*_~#>]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _fallback_title_from_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.path and parsed.path.strip("/"):
        return parsed.path.strip("/").split("/")[-1]
    return parsed.netloc or url


def _normalize_json_like_text(value: str) -> str:
    normalized = value.strip()
    fence_match = re.match(r"(?is)^```(?:json)?\s*(.*?)\s*```$", normalized)
    if fence_match:
        normalized = fence_match.group(1).strip()
    normalized = normalized.replace("\\\\_", "\\u005f")
    normalized = normalized.replace("\\_", "_")
    normalized = normalized.replace("\\-", "-")
    normalized = normalized.replace("\\*", "*")
    normalized = normalized.replace("\\`", "`")
    return normalized


def _decode_json_like_string(value: str) -> str:
    normalized = value.replace("\\\\_", "\\u005f")
    normalized = normalized.replace("\\_", "_")
    normalized = normalized.replace("\\-", "-")
    normalized = normalized.replace("\\*", "*")
    normalized = normalized.replace("\\`", "`")
    try:
        return json.loads(f'"{normalized}"')
    except json.JSONDecodeError:
        return normalized.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')


def _extract_json_like_string_field(text: str, key: str) -> str:
    pattern = rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"'
    match = re.search(pattern, text)
    if not match:
        return ""
    return _decode_json_like_string(match.group(1)).strip()


def _extract_json_like_int_field(text: str, key: str) -> int | None:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*(-?\d+)', text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _extract_json_like_bool_field(text: str, key: str) -> bool | None:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*(true|false)', text)
    if not match:
        return None
    return match.group(1) == "true"


def extract_reddit_json_post_fields(text: str) -> dict[str, object] | None:
    normalized_text = _normalize_json_like_text(text)
    try:
        payload = json.loads(normalized_text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, list) and payload:
        listing = payload[0] if isinstance(payload[0], dict) else {}
        children = (((listing.get("data") or {}).get("children")) or []) if isinstance(listing, dict) else []
        if children:
            first = children[0] if isinstance(children[0], dict) else {}
            post = first.get("data") if isinstance(first, dict) else None
            if isinstance(post, dict):
                return post

    fallback_text = normalized_text
    title = _extract_json_like_string_field(fallback_text, "title")
    subreddit = _extract_json_like_string_field(fallback_text, "subreddit")
    subreddit_name_prefixed = _extract_json_like_string_field(fallback_text, "subreddit_name_prefixed")
    author = _extract_json_like_string_field(fallback_text, "author")
    selftext = _extract_json_like_string_field(fallback_text, "selftext")
    permalink = _extract_json_like_string_field(fallback_text, "permalink")
    post_id = _extract_json_like_string_field(fallback_text, "id")
    removed_by_category = _extract_json_like_string_field(fallback_text, "removed_by_category")
    score = _extract_json_like_int_field(fallback_text, "score")
    num_comments = _extract_json_like_int_field(fallback_text, "num_comments")
    over_18 = _extract_json_like_bool_field(fallback_text, "over_18")

    if not any((title, subreddit, subreddit_name_prefixed, author, selftext, permalink, post_id)):
        return None

    return {
        "title": title,
        "subreddit": subreddit,
        "subreddit_name_prefixed": subreddit_name_prefixed,
        "author": author,
        "selftext": selftext,
        "permalink": permalink,
        "id": post_id,
        "removed_by_category": removed_by_category,
        "score": score,
        "num_comments": num_comments,
        "over_18": over_18,
    }


def _extract_reddit_json_post_markdown(url: str, text: str) -> str | None:
    post = extract_reddit_json_post_fields(text)
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
    suspicious_title_markers = (
        "reddit - the heart of the internet",
        "reddit - dive into anything",
    )
    if not title or title_lower in suspicious_title_markers:
        return None
    if removed_by_category in {"deleted", "removed"}:
        return None
    if str(author).strip().lower() == "[deleted]" and body in {"", "[removed]", "[deleted]"}:
        return None
    if over_18 is True:
        return None

    parts = [f"# {title}", ""]
    parts.append("## Thread Metadata")
    parts.append("")
    parts.append(f"- source_format: reddit_json")
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
    parts.append("")

    if body and body not in {"[removed]", "[deleted]"}:
        parts.append("## Post Body")
        parts.append("")
        parts.append(body)
        parts.append("")
    else:
        parts.append("> Reddit JSON endpoint exposed the thread metadata, but the post body is unavailable.")
        parts.append("")

    return "\n".join(parts).strip()


def infer_title(text: str, url: str) -> str:
    parsed = urlparse(url)
    domain = (parsed.netloc or "").lower()
    if domain == "reddit.com" or domain.endswith(".reddit.com"):
        if parsed.path.endswith(".json"):
            markdown = _extract_reddit_json_post_markdown(url, text)
            if markdown:
                for raw_line in markdown.splitlines():
                    line = raw_line.strip()
                    if line.startswith("# "):
                        title = _clean_markdown_text(line[2:])
                        if title:
                            return title
    if domain == "github.com" or domain.endswith(".github.com"):
        github_title_match = re.search(r"GitHub\s*-\s*([^:·\n]+(?:/[^:·\n]+)?)", text)
        if github_title_match:
            title = _clean_markdown_text(github_title_match.group(1))
            if title:
                return title
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("```") or _YAML_FIELD_PATTERN.match(line):
                continue
            if line.startswith("# "):
                title = _clean_markdown_text(line[2:])
                if title:
                    return title
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        if _YAML_FIELD_PATTERN.match(line):
            continue
        heading_match = _ATX_HEADING_PATTERN.match(line)
        if heading_match:
            title = _clean_markdown_text(heading_match.group(2))
            if title:
                return title
        if line.startswith(("---", "|", "- ", "* ", "> ", "http://", "https://", "<")):
            continue
        if len(line) > 120:
            continue
        title = _clean_markdown_text(line)
        if title:
            return title
    return _fallback_title_from_url(url)


def extract_heading_outline(text: str, max_items: int = _MAX_HEADING_ITEMS) -> list[str]:
    headings: list[str] = []
    for raw_line in text.splitlines():
        match = _ATX_HEADING_PATTERN.match(raw_line.strip())
        if not match:
            continue
        level = len(match.group(1))
        heading = _clean_markdown_text(match.group(2))
        if not heading:
            continue
        headings.append(f"H{level}: {heading}")
        if len(headings) >= max_items:
            break
    return headings


def _split_markdown_table_line(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not headers:
        return ""
    normalized_headers = [header or f"column_{index + 1}" for index, header in enumerate(headers)]
    separator = ["---"] * len(normalized_headers)
    lines = [
        "| " + " | ".join(normalized_headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows[:_MAX_TABLE_ROWS]:
        padded = row[: len(normalized_headers)] + [""] * max(0, len(normalized_headers) - len(row))
        lines.append("| " + " | ".join(padded[: len(normalized_headers)]) + " |")
    return "\n".join(lines)


def extract_markdown_tables(text: str, max_tables: int = _MAX_TABLE_COUNT) -> list[str]:
    tables: list[str] = []
    lines = text.splitlines()
    index = 0
    while index + 1 < len(lines) and len(tables) < max_tables:
        header_line = lines[index]
        separator_line = lines[index + 1]
        if "|" not in header_line or not _MARKDOWN_TABLE_SEPARATOR_PATTERN.match(separator_line):
            index += 1
            continue

        headers = _split_markdown_table_line(header_line)
        rows: list[list[str]] = []
        pointer = index + 2
        while pointer < len(lines):
            row_line = lines[pointer]
            if not row_line.strip() or "|" not in row_line:
                break
            rows.append(_split_markdown_table_line(row_line))
            pointer += 1

        rendered = _render_markdown_table(headers, rows)
        if rendered:
            tables.append(rendered)
        index = pointer

    return tables


def _strip_html(value: str) -> str:
    normalized = re.sub(r"(?is)<br\s*/?>", "\n", value)
    normalized = _HTML_TAG_PATTERN.sub("", normalized)
    normalized = html.unescape(normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _parse_html_table_block(block: str) -> str:
    rows: list[list[str]] = []
    header_row: list[str] | None = None
    for row_match in _HTML_ROW_PATTERN.finditer(block):
        cells: list[str] = []
        header_flags: list[bool] = []
        for cell_match in _HTML_CELL_PATTERN.finditer(row_match.group(1)):
            cells.append(_strip_html(cell_match.group(2)))
            header_flags.append(cell_match.group("tag").lower() == "h")
        if not cells:
            continue
        if header_row is None and any(header_flags):
            header_row = cells
            continue
        rows.append(cells)

    if header_row is None and rows:
        width = max(len(row) for row in rows)
        header_row = [f"column_{index + 1}" for index in range(width)]

    if not header_row:
        return ""
    return _render_markdown_table(header_row, rows)


def extract_html_tables(text: str, max_tables: int = _MAX_TABLE_COUNT) -> list[str]:
    blocks: list[str] = []
    blocks.extend(match.group(0) for match in _HTML_TABLE_PATTERN.finditer(text))
    if not blocks:
        blocks.extend(match.group(0) for match in _HTML_ROW_BLOCK_PATTERN.finditer(text))

    rendered_tables: list[str] = []
    seen: set[str] = set()
    for block in blocks:
        rendered = _parse_html_table_block(block)
        if not rendered or rendered in seen:
            continue
        seen.add(rendered)
        rendered_tables.append(rendered)
        if len(rendered_tables) >= max_tables:
            break
    return rendered_tables


def augment_fetched_markdown(url: str, text: str) -> str:
    content = (text or "").strip()
    if not content:
        return text
    parsed = urlparse(url)
    domain = (parsed.netloc or "").lower()
    if (domain == "reddit.com" or domain.endswith(".reddit.com")) and parsed.path.endswith(".json"):
        normalized = _extract_reddit_json_post_markdown(url, content)
        if normalized:
            content = normalized

    title = infer_title(content, url)
    headings = extract_heading_outline(content)
    tables = extract_markdown_tables(content)
    for table in extract_html_tables(content):
        if table not in tables:
            tables.append(table)
        if len(tables) >= _MAX_TABLE_COUNT:
            break

    sections = [
        "---",
        f"source_url: {url}",
        f"inferred_title: {title}",
        "fetch_structure_version: 1",
        "---",
        "",
        "## Extraction Aids",
        "",
        f"- source_url: {url}",
        f"- inferred_title: {title}",
        f"- heading_count: {len(headings)}",
        f"- normalized_table_count: {len(tables)}",
    ]

    if headings:
        sections.extend(["", "### Heading Outline", ""])
        sections.extend(f"- {heading}" for heading in headings)

    if tables:
        sections.extend(["", "### Normalized Tables", ""])
        for index, table in enumerate(tables, 1):
            sections.extend([f"#### Table {index}", "", table, ""])

    sections.extend(["", "## Original Content", "", content])
    return "\n".join(sections).strip() + "\n"
