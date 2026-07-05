"""Score InferTutor benchmark result JSON files."""

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table


console = Console()


def score(data: dict) -> float:
    config = data["config"]
    result = data["results"]
    users = config.get("users", 1)
    total_gpus = max(config.get("total_gpus", 1), 1)
    error_rate = result.get("error_rate", 1)
    goodput = result.get("aggregate_stream_chunks_per_s", 0) * max(0, 1 - error_rate)
    ttft = max(result.get("ttft_p95_ms", 1) / 1000, 0.001)
    itl = max(result.get("itl_p95_ms", 1) / 1000, 0.001)
    return goodput * users / (ttft * itl * total_gpus)


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        console.print("Usage: python score_infertutor.py [results_file_or_directory]")
        return

    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results_infertutor")
    if not path.exists():
        console.print(f"[red]No result path found:[/red] {path}")
        return

    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not files:
        console.print(f"[red]No JSON results found in:[/red] {path}")
        return

    rows = []
    for file in files:
        data = json.loads(file.read_text())
        rows.append((score(data), file, data))
    rows.sort(reverse=True, key=lambda x: x[0])

    table = Table(title="InferTutor Leaderboard")
    for col in ["file", "mode", "users", "gpus", "err%", "TTFT p95", "ITL p95", "throughput", "score"]:
        table.add_column(col, justify="right" if col != "file" else "left")

    for value, file, data in rows:
        c = data["config"]
        r = data["results"]
        table.add_row(
            file.name,
            c.get("mode", ""),
            str(c.get("users", "")),
            str(c.get("total_gpus", 1)),
            f'{100 * r.get("error_rate", 0):.1f}',
            f'{r.get("ttft_p95_ms", 0):.0f} ms',
            f'{r.get("itl_p95_ms", 0):.1f} ms',
            f'{r.get("aggregate_stream_chunks_per_s", 0):.1f}',
            f"{value:.0f}",
        )

    console.print(table)


if __name__ == "__main__":
    main()
