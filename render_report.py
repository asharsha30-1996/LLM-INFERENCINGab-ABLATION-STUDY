"""Render EXPERIMENT_REPORT.md to HTML.

This intentionally uses only the Python standard library so the report can be
rendered on the workshop machine without installing Pandoc or Markdown extras.
"""

from __future__ import annotations

import html
import re
from pathlib import Path


ROOT = Path(__file__).parent
SOURCE = ROOT / "EXPERIMENT_REPORT.md"
TARGET = ROOT / "EXPERIMENT_REPORT.html"
BLOCK_LABELS = {
    "Purpose:",
    "Implementation:",
    "Command:",
    "Commands:",
    "Result file:",
    "Route counts:",
    "Metrics:",
    "Interpretation:",
    "Result:",
    "Observed failure:",
    "Small feasibility command:",
    "Key Finding:",
    "Decision:",
    "Cost note:",
    "Final best result:",
    "After each experiment:",
}


def is_narrative_label(value: str) -> bool:
    """Return true for short experiment/config labels that read better as bold."""

    code_markers = [
        "/",
        "\\",
        ".json",
        ".py",
        ".md",
        ".html",
        ".pdf",
        "://",
        "--",
        "{",
        "}",
        "(",
        ")",
        "=",
        "_",
        " ",
    ]
    if any(marker in value for marker in code_markers):
        return False
    if len(value) > 32:
        return False
    if re.fullmatch(r"(b|seq|u|r|ci|fp|fp8|tp)\d+[a-z0-9-]*", value, re.IGNORECASE):
        return True
    if re.fullmatch(r"[a-z]+(?:-[a-z0-9]+){1,5}", value, re.IGNORECASE):
        return True
    if re.fullmatch(r"\d+(?:-\d+)?\s*(?:users?|replicas?|GPUs?|H100s?)", value, re.IGNORECASE):
        return True
    return False


def inline(text: str) -> str:
    escaped = html.escape(text)

    def render_code(match: re.Match[str]) -> str:
        value = match.group(1)
        tag = "strong" if is_narrative_label(value) else "code"
        return f"<{tag}>{value}</{tag}>"

    escaped = re.sub(r"`([^`]+)`", render_code, escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def flush_paragraph(parts: list[str], out: list[str]) -> None:
    if parts:
        out.append(f"<p>{inline(' '.join(parts))}</p>")
        parts.clear()


def flush_list(items: list[str], out: list[str]) -> None:
    if items:
        out.append("<ul>")
        out.extend(f"<li>{inline(item)}</li>" for item in items)
        out.append("</ul>")
        items.clear()


def parse_table(lines: list[str], start: int, out: list[str]) -> int:
    rows: list[list[str]] = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        row = [cell.strip() for cell in lines[i].strip().strip("|").split("|")]
        rows.append(row)
        i += 1

    if len(rows) < 2:
        return start

    header = rows[0]
    body = rows[2:] if all(set(cell) <= {"-", ":"} for cell in rows[1]) else rows[1:]
    out.append("<table>")
    out.append("<thead><tr>" + "".join(f"<th>{inline(cell)}</th>" for cell in header) + "</tr></thead>")
    out.append("<tbody>")
    for row in body:
        out.append("<tr>" + "".join(f"<td>{inline(cell)}</td>" for cell in row) + "</tr>")
    out.append("</tbody></table>")
    return i


def render(markdown: str) -> str:
    lines = markdown.splitlines()
    out: list[str] = []
    paragraph: list[str] = []
    bullets: list[str] = []
    in_code = False
    code_lines: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph(paragraph, out)
            flush_list(bullets, out)
            if in_code:
                out.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
                code_lines.clear()
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if not stripped:
            flush_paragraph(paragraph, out)
            flush_list(bullets, out)
            i += 1
            continue

        if stripped.startswith("|"):
            flush_paragraph(paragraph, out)
            flush_list(bullets, out)
            i = parse_table(lines, i, out)
            continue

        image = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if image:
            flush_paragraph(paragraph, out)
            flush_list(bullets, out)
            alt = html.escape(image.group(1))
            src = html.escape(image.group(2))
            out.append(f'<figure><img src="{src}" alt="{alt}"><figcaption>{alt}</figcaption></figure>')
            i += 1
            continue

        if stripped in BLOCK_LABELS:
            flush_paragraph(paragraph, out)
            flush_list(bullets, out)
            out.append(f'<div class="block-label">{inline(stripped[:-1])}</div>')
            i += 1
            continue

        inline_label = next((label for label in sorted(BLOCK_LABELS, key=len, reverse=True) if stripped.startswith(label + " ")), None)
        if inline_label:
            flush_paragraph(paragraph, out)
            flush_list(bullets, out)
            out.append(f'<div class="block-label">{inline(inline_label[:-1])}</div>')
            rest = stripped[len(inline_label) :].strip()
            if rest:
                out.append(f"<p>{inline(rest)}</p>")
            i += 1
            continue

        header = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if header:
            flush_paragraph(paragraph, out)
            flush_list(bullets, out)
            level = len(header.group(1))
            out.append(f"<h{level}>{inline(header.group(2))}</h{level}>")
            i += 1
            continue

        if stripped.startswith("- "):
            flush_paragraph(paragraph, out)
            bullets.append(stripped[2:])
            i += 1
            continue

        paragraph.append(stripped)
        i += 1

    flush_paragraph(paragraph, out)
    flush_list(bullets, out)

    body = "\n".join(out)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>InferTutor Arena Experiment Report</title>
  <style>
    @page {{ margin: 0.65in; }}
    body {{
      color: #1f2933;
      font-family: "Segoe UI", Arial, sans-serif;
      font-size: 10.5pt;
      line-height: 1.45;
      margin: 0 auto;
      max-width: 980px;
    }}
    h1 {{ font-size: 25pt; border-bottom: 2px solid #1f2933; padding-bottom: 8px; }}
    h2 {{ font-size: 17pt; margin-top: 28px; border-bottom: 1px solid #c9d2dc; padding-bottom: 4px; }}
    h3 {{ font-size: 13pt; margin-top: 22px; color: #243b53; }}
    p {{ margin: 8px 0; }}
    code {{ background: #eef2f7; border-radius: 3px; padding: 1px 4px; font-family: Consolas, monospace; font-size: 9.3pt; }}
    pre {{ background: #0f172a; color: #e2e8f0; border-radius: 6px; padding: 10px; white-space: pre-wrap; overflow-wrap: anywhere; }}
    pre code {{ background: transparent; color: inherit; padding: 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0 18px; font-size: 8.8pt; page-break-inside: avoid; }}
    th, td {{ border: 1px solid #c9d2dc; padding: 5px 6px; vertical-align: top; }}
    th {{ background: #edf2f7; font-weight: 700; }}
    tr:nth-child(even) td {{ background: #f8fafc; }}
    .block-label {{
      background: #edf2f7;
      border-left: 4px solid #2563eb;
      color: #1f2933;
      font-size: 9.4pt;
      font-weight: 700;
      letter-spacing: 0.02em;
      margin: 14px 0 6px;
      padding: 5px 8px;
      text-transform: uppercase;
    }}
    figure {{ margin: 14px 0 22px; page-break-inside: avoid; }}
    figure img {{ display: block; max-width: 100%; height: auto; border: 1px solid #d8e0e8; border-radius: 6px; }}
    figcaption {{ color: #52616f; font-size: 8.8pt; margin-top: 5px; text-align: center; }}
    ul {{ margin-top: 6px; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> None:
    TARGET.write_text(render(SOURCE.read_text(encoding="utf-8")), encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
