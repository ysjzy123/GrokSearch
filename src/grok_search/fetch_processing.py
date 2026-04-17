from __future__ import annotations

import html
import re
from urllib.parse import urlparse


_ATX_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
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


def infer_title(text: str, url: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
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
