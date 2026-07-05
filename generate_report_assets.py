"""Generate SVG plots and architecture diagrams for EXPERIMENT_REPORT.md."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path


ROOT = Path(__file__).parent
RESULTS = ROOT / "starter_code" / "results_infertutor"
ASSETS = ROOT / "report_assets"


def score(config: dict, results: dict) -> float:
    users = config["users"]
    total_gpus = max(config.get("total_gpus", 1), 1)
    err = results["error_rate"]
    goodput = results["aggregate_stream_chunks_per_s"] * (1 - err)
    ttft = max(results["ttft_p95_ms"] / 1000, 1e-6)
    itl = max(results["itl_p95_ms"] / 1000, 1e-6)
    return goodput * users / (ttft * itl * total_gpus)


def load_result(filename: str) -> dict:
    data = json.loads((RESULTS / filename).read_text(encoding="utf-8"))
    data["filename"] = filename
    data["score"] = score(data["config"], data["results"])
    return data


def svg(width: int, height: int, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="{width}" height="{height}" fill="#ffffff"/>
<style>
.title {{ font: 700 22px Segoe UI, Arial, sans-serif; fill: #1f2933; }}
.subtitle {{ font: 13px Segoe UI, Arial, sans-serif; fill: #52616f; }}
.label {{ font: 12px Segoe UI, Arial, sans-serif; fill: #243b53; }}
.small {{ font: 10px Segoe UI, Arial, sans-serif; fill: #52616f; }}
.axis {{ stroke: #9fb3c8; stroke-width: 1; }}
.grid {{ stroke: #e4ebf2; stroke-width: 1; }}
.box {{ fill: #f8fafc; stroke: #b8c7d6; stroke-width: 1.5; rx: 8; }}
.box2 {{ fill: #edf7f6; stroke: #4f9d96; stroke-width: 1.5; rx: 8; }}
.box3 {{ fill: #fff7ed; stroke: #d97706; stroke-width: 1.5; rx: 8; }}
.arrow {{ stroke: #52616f; stroke-width: 2; fill: none; marker-end: url(#arrow); }}
</style>
<defs>
<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
<path d="M0,0 L0,6 L9,3 z" fill="#52616f"/>
</marker>
</defs>
{body}
</svg>
"""


def write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def text(x: float, y: float, content: str, klass: str = "label", anchor: str = "start") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{klass}" text-anchor="{anchor}">{html.escape(content)}</text>'


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", rx: int = 4) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'


def line(x1: float, y1: float, x2: float, y2: float, klass: str = "axis") -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{klass}"/>'


def score_chart() -> None:
    files = [
        "baseline-mixed-r2_mixed_100u_1783186000.json",
        "mixed-r4-b2048-u120_mixed_120u_1783186496.json",
        "mixed-r4-b4096-u120_mixed_120u_1783188479.json",
        "mixed-r4-b1024-u120_mixed_120u_1783190482.json",
        "mixed-r4-b2048-u120-noprefix_mixed_120u_1783191286.json",
        "mixed-r4-b2048-u120-nochunk_mixed_120u_1783191838.json",
        "mixed-r4-b2048-ngram4-u120-r3_mixed_120u_1783196706.json",
        "mixed-r4-b2048-fp8w-u120_mixed_120u_1783197169.json",
        "routed-text2-heavy2-u120_routed_120u_1783198666.json",
        "routed-text1-heavy3-u120_routed_120u_1783269742.json",
        "mixed-r4-b2048-compiled-u120_mixed_120u_1783199261.json",
    ]
    labels = [
        "2-rep baseline",
        "Best b2048",
        "b4096",
        "b1024",
        "No prefix",
        "No chunk",
        "N-gram spec",
        "FP8 weights",
        "Routed 2+2",
        "Routed 1+3",
        "Compiled",
    ]
    rows = list(zip(labels, [load_result(f) for f in files]))
    width, height = 1120, 560
    left, top, plot_w, row_h = 220, 92, 820, 36
    max_score = max(r["score"] for _, r in rows)
    body = [
        text(36, 38, "Score Comparison Across Key Runs", "title"),
        text(36, 62, "Higher is better. Best 4-replica eager mixed configuration stays ahead of advanced variants.", "subtitle"),
    ]
    for i in range(0, 5):
        x = left + plot_w * i / 4
        body.append(line(x, top - 8, x, top + row_h * len(rows), "grid"))
        body.append(text(x, top + row_h * len(rows) + 24, f"{max_score * i / 4 / 1_000_000:.1f}M", "small", "middle"))
    for idx, (label, result) in enumerate(rows):
        y = top + idx * row_h
        val = result["score"]
        bar_w = plot_w * val / max_score
        fill = "#1f77b4" if label != "Best b2048" else "#16a34a"
        body.append(text(left - 12, y + 22, label, "label", "end"))
        body.append(rect(left, y + 7, bar_w, 20, fill, rx=5))
        body.append(text(left + bar_w + 8, y + 22, f"{val / 1_000_000:.2f}M", "small"))
    write(ASSETS / "score_comparison.svg", svg(width, height, "\n".join(body)))


def latency_throughput_chart() -> None:
    files = [
        "baseline-mixed-r2_mixed_100u_1783186000.json",
        "mixed-r4-b2048-u120_mixed_120u_1783186496.json",
        "mixed-r4-b4096-u120_mixed_120u_1783188479.json",
        "mixed-r4-b1024-u120_mixed_120u_1783190482.json",
        "mixed-r4-b2048-u120-noprefix_mixed_120u_1783191286.json",
        "mixed-r4-b2048-u120-nochunk_mixed_120u_1783191838.json",
        "mixed-r4-b2048-seq16-u120_mixed_120u_1783192232.json",
        "mixed-r4-b2048-seq48-u120_mixed_120u_1783193652.json",
        "mixed-r4-b2048-u130_mixed_130u_1783193103.json",
        "mixed-r4-b2048-u140_mixed_140u_1783192699.json",
        "routed-text1-heavy3-u120_routed_120u_1783269742.json",
        "mixed-r4-b2048-compiled-u120_mixed_120u_1783199261.json",
    ]
    labels = ["baseline", "best", "b4096", "b1024", "no-prefix", "no-chunk", "seq16", "seq48", "u130", "u140", "routed 1+3", "compiled"]
    points = list(zip(labels, [load_result(f) for f in files]))
    width, height = 1120, 640
    left, top, plot_w, plot_h = 92, 90, 940, 445
    max_x = 9500
    max_y = 3500

    def xmap(ttft: float) -> float:
        return left + plot_w * ttft / max_x

    def ymap(tput: float) -> float:
        return top + plot_h - plot_h * tput / max_y

    body = [
        text(36, 38, "Latency/Throughput Tradeoff", "title"),
        text(36, 62, "The best run sits in the low-TTFT, high-throughput corner. Saturated runs drift right and down.", "subtitle"),
    ]
    for i in range(0, 6):
        x = left + plot_w * i / 5
        y = top + plot_h * i / 5
        body.append(line(x, top, x, top + plot_h, "grid"))
        body.append(line(left, y, left + plot_w, y, "grid"))
        body.append(text(x, top + plot_h + 24, f"{max_x * i / 5 / 1000:.1f}s", "small", "middle"))
        body.append(text(left - 10, y + 4, f"{max_y * (5 - i) / 5:.0f}", "small", "end"))
    body.append(line(left, top + plot_h, left + plot_w, top + plot_h))
    body.append(line(left, top, left, top + plot_h))
    body.append(text(left + plot_w / 2, top + plot_h + 52, "TTFT p95", "label", "middle"))
    body.append(text(22, top + plot_h / 2, "Throughput", "label", "middle"))
    for label, result in points:
        r = result["results"]
        x = xmap(min(r["ttft_p95_ms"], max_x))
        y = ymap(min(r["aggregate_stream_chunks_per_s"], max_y))
        radius = 8 + 12 * math.sqrt(result["score"] / max(p["score"] for _, p in points))
        fill = "#16a34a" if label == "best" else "#2563eb" if label == "compiled" else "#f97316" if label in {"u130", "u140"} else "#64748b"
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" fill-opacity="0.78" stroke="#1f2933" stroke-width="1"/>')
        body.append(text(x + 12, y - 8, label, "small"))
    write(ASSETS / "latency_throughput.svg", svg(width, height, "\n".join(body)))


def architecture_diagram() -> None:
    width, height = 1120, 520
    body = [
        text(36, 38, "Final Monolithic Serving Architecture", "title"),
        text(36, 62, "Four Modal replicas each run one vLLM server that performs prefill and decode in the same scheduler.", "subtitle"),
        '<rect x="62" y="118" width="190" height="96" class="box"/>',
        text(157, 154, "Load Tester", "label", "middle"),
        text(157, 178, "mixed prompts", "small", "middle"),
        '<path d="M252 166 C320 166, 340 166, 400 166" class="arrow"/>',
        '<rect x="400" y="96" width="660" height="292" class="box"/>',
        text(730, 126, "Modal App: 4 H100 Replicas", "label", "middle"),
    ]
    x0, y0, w, h = 440, 160, 130, 150
    for idx in range(4):
        x = x0 + idx * 150
        body.append(f'<rect x="{x}" y="{y0}" width="{w}" height="{h}" class="box2"/>')
        body.append(text(x + w / 2, y0 + 30, f"Replica {idx + 1}", "label", "middle"))
        body.append(text(x + w / 2, y0 + 58, "vLLM server", "small", "middle"))
        body.append(text(x + w / 2, y0 + 84, "prefill + decode", "small", "middle"))
        body.append(text(x + w / 2, y0 + 110, "b2048 / seq32", "small", "middle"))
    body.extend(
        [
            '<path d="M730 388 C730 424, 730 424, 730 454" class="arrow"/>',
            '<rect x="574" y="454" width="312" height="44" class="box3"/>',
            text(730, 482, "Best score: 7.08M, 0.0% errors", "label", "middle"),
        ]
    )
    write(ASSETS / "architecture_monolithic.svg", svg(width, height, "\n".join(body)))


def routed_architecture_diagram() -> None:
    width, height = 1120, 600
    body = [
        text(36, 38, "Request-Class Routing Prototype", "title"),
        text(36, 62, "The mixed workload was split into a text pool and a heavy image/long-context pool under the same 4-GPU budget.", "subtitle"),
        '<rect x="48" y="132" width="170" height="92" class="box"/>',
        text(133, 166, "Load Tester", "label", "middle"),
        text(133, 190, "official mixed", "small", "middle"),
        '<path d="M218 178 C266 178, 286 178, 326 178" class="arrow"/>',
        '<rect x="326" y="124" width="176" height="108" class="box"/>',
        text(414, 158, "Routing Client", "label", "middle"),
        text(414, 183, "text vs heavy", "small", "middle"),
        text(414, 205, "same scoring path", "small", "middle"),
        '<path d="M502 152 C560 110, 600 104, 650 104" class="arrow"/>',
        '<path d="M502 204 C560 280, 600 302, 650 302" class="arrow"/>',
        '<rect x="650" y="76" width="384" height="142" class="box2"/>',
        text(842, 110, "Text Endpoint", "label", "middle"),
        text(842, 135, "2 H100 replicas", "small", "middle"),
        text(842, 158, "short text prompts", "small", "middle"),
        text(842, 181, "max-batch-tokens 4096", "small", "middle"),
        '<rect x="694" y="226" width="296" height="28" fill="#eefaf7" stroke="#91cbbf" rx="6"/>',
        text(842, 245, "994 text requests, 1 error", "small", "middle"),
        '<rect x="650" y="286" width="384" height="162" class="box3"/>',
        text(842, 320, "Heavy Endpoint", "label", "middle"),
        text(842, 345, "2 H100 replicas", "small", "middle"),
        text(842, 368, "long + image prompts", "small", "middle"),
        text(842, 391, "max-batch-tokens 2048", "small", "middle"),
        '<rect x="694" y="458" width="296" height="28" fill="#fff1e6" stroke="#f4a261" rx="6"/>',
        text(842, 477, "311 long + 437 image requests", "small", "middle"),
        '<rect x="186" y="502" width="748" height="58" fill="#fef2f2" stroke="#dc2626" rx="8"/>',
        text(560, 526, "Observed bottleneck", "label", "middle"),
        text(560, 548, "The heavy pool had only 2 replicas, so aggregate TTFT p95 rose to 5999.6 ms and score fell to 235,765.", "small", "middle"),
    ]
    write(ASSETS / "architecture_routed.svg", svg(width, height, "\n".join(body)))


def routed_asymmetric_architecture_diagram() -> None:
    width, height = 1120, 600
    body = [
        text(36, 38, "Asymmetric Request-Class Routing Prototype", "title"),
        text(36, 62, "The 1+3 routing variant keeps the 4-GPU budget but shifts capacity toward long/image traffic.", "subtitle"),
        '<rect x="48" y="132" width="170" height="92" class="box"/>',
        text(133, 166, "Load Tester", "label", "middle"),
        text(133, 190, "120 mixed users", "small", "middle"),
        '<path d="M218 178 C266 178, 286 178, 326 178" class="arrow"/>',
        '<rect x="326" y="124" width="176" height="108" class="box"/>',
        text(414, 158, "Routing Client", "label", "middle"),
        text(414, 183, "text -> r1", "small", "middle"),
        text(414, 205, "long/image -> r3", "small", "middle"),
        '<path d="M502 152 C560 110, 600 104, 650 104" class="arrow"/>',
        '<path d="M502 204 C560 280, 600 302, 650 302" class="arrow"/>',
        '<rect x="650" y="76" width="384" height="142" class="box2"/>',
        text(842, 110, "Text Endpoint", "label", "middle"),
        text(842, 135, "1 H100 replica", "small", "middle"),
        text(842, 158, "1208 text requests", "small", "middle"),
        text(842, 181, "0 text errors", "small", "middle"),
        '<rect x="650" y="286" width="384" height="162" class="box3"/>',
        text(842, 320, "Heavy Endpoint", "label", "middle"),
        text(842, 345, "3 H100 replicas", "small", "middle"),
        text(842, 368, "392 long + 532 image requests", "small", "middle"),
        text(842, 391, "0 long/image errors", "small", "middle"),
        '<rect x="188" y="502" width="744" height="58" fill="#eefaf7" stroke="#2f9e44" rx="8"/>',
        text(560, 526, "Observed result", "label", "middle"),
        text(560, 548, "Score improved to 621,303 versus 235,765 for 2+2 routing, but remained below the monolithic best.", "small", "middle"),
    ]
    write(ASSETS / "architecture_routed_asymmetric.svg", svg(width, height, "\n".join(body)))


def disagg_diagram() -> None:
    width, height = 1120, 560
    body = [
        text(36, 38, "Attempted Disaggregated Prefill/Decode Architecture", "title"),
        text(36, 62, "A prefiller/decoder pair was attempted with LMCache KV transfer, but engine initialization failed.", "subtitle"),
        '<rect x="52" y="120" width="165" height="86" class="box"/>',
        text(134, 154, "Load Tester", "label", "middle"),
        text(134, 178, "OpenAI API", "small", "middle"),
        '<path d="M217 164 C270 164, 288 164, 330 164" class="arrow"/>',
        '<rect x="330" y="112" width="170" height="104" class="box"/>',
        text(415, 150, "FastAPI Proxy", "label", "middle"),
        text(415, 176, "prefill then stream", "small", "middle"),
        '<path d="M500 146 C560 116, 594 110, 650 110" class="arrow"/>',
        '<path d="M500 182 C560 214, 594 226, 650 226" class="arrow"/>',
        '<rect x="650" y="72" width="220" height="112" class="box2"/>',
        text(760, 112, "Prefiller vLLM", "label", "middle"),
        text(760, 138, "GPU 0", "small", "middle"),
        text(760, 160, "KV producer", "small", "middle"),
        '<rect x="650" y="204" width="220" height="112" class="box2"/>',
        text(760, 244, "Decoder vLLM", "label", "middle"),
        text(760, 270, "GPU 1", "small", "middle"),
        text(760, 292, "KV consumer", "small", "middle"),
        '<path d="M760 184 C760 194, 760 194, 760 204" class="arrow"/>',
        '<rect x="916" y="128" width="158" height="132" class="box3"/>',
        text(995, 168, "LMCache", "label", "middle"),
        text(995, 194, "NIXL KV", "small", "middle"),
        text(995, 216, "transfer", "small", "middle"),
        '<path d="M870 128 C900 128, 900 172, 916 172" class="arrow"/>',
        '<path d="M916 218 C900 218, 900 260, 870 260" class="arrow"/>',
        '<rect x="222" y="384" width="696" height="90" fill="#fef2f2" stroke="#dc2626" rx="8"/>',
        text(570, 420, "Observed blocker", "label", "middle"),
        text(570, 448, "LMCache imports cdiv from vLLM utils, but vLLM 0.21.0 no longer exposes it.", "small", "middle"),
    ]
    write(ASSETS / "architecture_disaggregated.svg", svg(width, height, "\n".join(body)))


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    score_chart()
    latency_throughput_chart()
    architecture_diagram()
    routed_architecture_diagram()
    routed_asymmetric_architecture_diagram()
    disagg_diagram()
    for path in sorted(ASSETS.glob("*.svg")):
        print(path)


if __name__ == "__main__":
    main()
