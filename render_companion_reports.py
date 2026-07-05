"""Render companion Markdown reports to HTML using the main report renderer."""

from __future__ import annotations

from pathlib import Path

from render_report import render


ROOT = Path(__file__).parent
REPORTS = [
    ("DISAGGREGATED_SERVING_DETAILS.md", "DISAGGREGATED_SERVING_DETAILS.html", "Disaggregated Serving Deep Dive"),
    ("ARCHITECTURE_NOVEL_METHODS.md", "ARCHITECTURE_NOVEL_METHODS.html", "Architecture and Novel Methods Summary"),
]


def main() -> None:
    for source_name, target_name, title in REPORTS:
        source = ROOT / source_name
        target = ROOT / target_name
        markdown = source.read_text(encoding="utf-8-sig")
        target.write_text(render(markdown, title=title), encoding="utf-8")
        print(target)


if __name__ == "__main__":
    main()