"""Routed workload load tester for InferTutor Arena.

This keeps the official mixed workload shape but sends short text prompts to a
text endpoint and long/image prompts to a heavy multimodal endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass, field

import httpx
from rich.console import Console
from rich.live import Live

from load_test_infertutor import IMAGE_URL, PROMPTS, ROOT, Stats


console = Console()


def choose_routed_messages() -> tuple[str, list[dict]]:
    """Build one mixed-workload request and return its route class."""

    system = {"role": "system", "content": PROMPTS["system_prompt"]}
    roll = random.random()
    if roll < 0.25:
        content = [
            {"type": "image_url", "image_url": {"url": IMAGE_URL}},
            {"type": "text", "text": random.choice(PROMPTS["image"])},
        ]
        return "image", [system, {"role": "user", "content": content}]
    if roll < 0.45:
        return "long", [system, {"role": "user", "content": random.choice(PROMPTS["long"])}]
    return "text", [system, {"role": "user", "content": random.choice(PROMPTS["text"])}]


@dataclass
class RouteCounts:
    sent: dict[str, int] = field(default_factory=lambda: {"text": 0, "long": 0, "image": 0})
    errors: dict[str, int] = field(default_factory=lambda: {"text": 0, "long": 0, "image": 0})
    samples: list[dict] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def record_sent(self, route: str):
        async with self.lock:
            self.sent[route] += 1

    async def record_error(self, route: str):
        async with self.lock:
            self.errors[route] += 1

    async def record_sample(self, sample: dict):
        async with self.lock:
            if len(self.samples) < 12:
                self.samples.append(sample)


def route_url(route: str, args) -> str:
    if route == "text":
        return args.text_url
    return args.heavy_url


async def check_endpoint_health(args):
    async with httpx.AsyncClient(timeout=20) as client:
        for name, url in [("text", args.text_url), ("heavy", args.heavy_url)]:
            health_url = f"{url.rstrip('/')}/health"
            try:
                resp = await client.get(health_url)
                console.print(f"[bold]{name} health[/bold] {health_url} -> HTTP {resp.status_code}")
                if resp.status_code != 200:
                    console.print(resp.text[:500], style="yellow")
            except Exception as exc:
                console.print(f"[red]{name} health failed[/red] {health_url}: {exc}")


async def user_loop(
    user_id: int,
    args,
    stats: Stats,
    route_counts: RouteCounts,
    stop_event: asyncio.Event,
):
    async with httpx.AsyncClient(timeout=args.request_timeout) as client:
        while not stop_event.is_set():
            route, messages = choose_routed_messages()
            await route_counts.record_sent(route)
            payload = {
                "model": args.model,
                "messages": messages,
                "max_tokens": args.max_tokens,
                "temperature": 0.2,
                "stream": True,
            }

            request_start = time.perf_counter()
            first_chunk_at = None
            chunk_times = []
            chunks = 0

            try:
                async with client.stream(
                    "POST",
                    f"{route_url(route, args).rstrip('/')}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    if resp.status_code != 200:
                        await stats.error()
                        await route_counts.record_error(route)
                        body = (await resp.aread()).decode("utf-8", errors="replace")
                        await route_counts.record_sample(
                            {
                                "route": route,
                                "url": route_url(route, args),
                                "status_code": resp.status_code,
                                "body": body[:1000],
                            }
                        )
                        continue

                    async for line in resp.aiter_lines():
                        if not line or line == "data: [DONE]":
                            continue
                        if line.startswith("data: "):
                            line = line[6:]
                        try:
                            chunk = json.loads(line)
                            content = chunk["choices"][0]["delta"].get("content", "")
                        except Exception:
                            continue
                        if content:
                            now = time.perf_counter()
                            first_chunk_at = first_chunk_at or now
                            chunk_times.append(now)
                            chunks += 1

                request_end = time.perf_counter()
                if first_chunk_at is None or chunks == 0:
                    await stats.error()
                    await route_counts.record_error(route)
                    continue

                gaps = [b - a for a, b in zip(chunk_times, chunk_times[1:])]
                ttft = (first_chunk_at - request_start) * 1000
                itl = (sum(gaps) / len(gaps) * 1000) if gaps else 0.0
                latency = (request_end - request_start) * 1000
                await stats.success(ttft, itl, latency, chunks)
            except Exception as exc:
                await stats.error()
                await route_counts.record_error(route)
                await route_counts.record_sample(
                    {
                        "route": route,
                        "url": route_url(route, args),
                        "exception": repr(exc),
                    }
                )

            await asyncio.sleep(random.uniform(args.min_pause, args.max_pause))


async def run(args):
    stats = Stats(started_at=time.time())
    route_counts = RouteCounts()
    stop_event = asyncio.Event()
    tasks = []
    await check_endpoint_health(args)

    async def ramp_users():
        delay = args.ramp_up / max(args.users, 1) if args.ramp_up else 0
        for i in range(args.users):
            if stop_event.is_set():
                return
            tasks.append(asyncio.create_task(user_loop(i, args, stats, route_counts, stop_event)))
            stats.active_users = i + 1
            if delay:
                await asyncio.sleep(delay)

    ramp_task = asyncio.create_task(ramp_users())
    with Live(stats.table(), refresh_per_second=0.5, console=console) as live:
        end = time.time() + args.duration
        while time.time() < end:
            await asyncio.sleep(2)
            live.update(stats.table())

    stop_event.set()
    ramp_task.cancel()
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    result = {
        "config": vars(args),
        "route_counts": {
            "sent": route_counts.sent,
            "errors": route_counts.errors,
        },
        "error_samples": route_counts.samples,
        "results": stats.results(),
    }
    out_dir = ROOT / "results_infertutor"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{args.label}_routed_{args.users}u_{int(time.time())}.json"
    out_file.write_text(json.dumps(result, indent=2))
    console.print(stats.table())
    if route_counts.samples:
        console.print("[yellow]First routed error samples:[/yellow]")
        console.print(json.dumps(route_counts.samples, indent=2))
    console.print(f"[green]Saved {out_file}[/green]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-url", required=True)
    parser.add_argument("--heavy-url", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--users", type=int, default=120)
    parser.add_argument("--duration", type=int, default=90)
    parser.add_argument("--ramp-up", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--min-pause", type=float, default=0.2)
    parser.add_argument("--max-pause", type=float, default=1.2)
    parser.add_argument("--label", default="routed")
    parser.add_argument("--total-gpus", type=int, default=4)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
