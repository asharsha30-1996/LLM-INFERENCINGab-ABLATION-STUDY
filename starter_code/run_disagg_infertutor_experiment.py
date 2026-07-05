"""
Deploy and benchmark the experimental disaggregated-prefill InferTutor app.

This runner is intentionally separate from run_infertutor_experiment.py because
the disaggregated app uses 2 GPUs per Modal container: one prefiller GPU and
one decoder GPU. With --replicas 2, the benchmark uses 4 total GPUs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx
from rich.console import Console
from rich.panel import Panel


console = Console()
ROOT = Path(__file__).parent
TEMPLATE = ROOT / "modal_disagg_infertutor_app.py"
GENERATED = ROOT / "modal_disagg_infertutor_app_generated.py"


def patch_modal_app(args) -> Path:
    source = TEMPLATE.read_text()
    replacements = {
        'MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-4B-Instruct")': f'MODEL_NAME = "{args.model}"',
        'GPU_TYPE = os.environ.get("GPU_TYPE", "H100")': f'GPU_TYPE = "{args.gpu_type}"',
        'DTYPE = os.environ.get("DTYPE", "bfloat16")': f'DTYPE = "{args.dtype}"',
        'MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))': f"MAX_MODEL_LEN = {args.max_model_len}",
        'MAX_NUM_BATCHED_TOKENS = int(os.environ.get("MAX_NUM_BATCHED_TOKENS", "2048"))': f"MAX_NUM_BATCHED_TOKENS = {args.max_batch_tokens}",
        'MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "32"))': f"MAX_NUM_SEQS = {args.max_seqs}",
        'CONCURRENT_INPUTS = int(os.environ.get("CONCURRENT_INPUTS", "64"))': f"CONCURRENT_INPUTS = {args.concurrent_inputs}",
        'MIN_CONTAINERS = int(os.environ.get("MIN_CONTAINERS", "2"))': f"MIN_CONTAINERS = {args.replicas}",
        'MAX_CONTAINERS = int(os.environ.get("MAX_CONTAINERS", "2"))': f"MAX_CONTAINERS = {args.replicas}",
        'MM_MAX_PIXELS = int(os.environ.get("MM_MAX_PIXELS", str(512 * 28 * 28)))': f"MM_MAX_PIXELS = {args.mm_max_pixels}",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)

    app_name = f"infertutor-{args.label}".replace("_", "-")
    source = source.replace('app = modal.App("infertutor-disagg-pd")', f'app = modal.App("{app_name}")')
    GENERATED.write_text(source)
    return GENERATED


def deploy(app_path: Path) -> str:
    proc = subprocess.run(
        ["modal", "deploy", str(app_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    console.print(proc.stdout)
    if proc.returncode != 0:
        console.print(proc.stderr, style="red")
        raise SystemExit(proc.returncode)

    normalized = re.sub(r"\s+", "", proc.stdout)
    match = re.search(r"https://[^\"'<>]+?modal\.run", normalized)
    if not match:
        raise RuntimeError("Could not find Modal endpoint in deploy output")
    return match.group(0)


def wait_for_health(url: str, timeout_s: int = 1200):
    console.print(f"[bold]Waiting for disaggregated vLLM health:[/bold] {url}/health")
    deadline = time.time() + timeout_s
    last_error = ""
    with httpx.Client(timeout=20) as client:
        while time.time() < deadline:
            try:
                resp = client.get(f"{url.rstrip('/')}/health")
                if resp.status_code == 200:
                    console.print("[green]Endpoint is healthy[/green]")
                    return
                last_error = f"HTTP {resp.status_code}: {resp.text[:2000]}"
            except Exception as exc:
                last_error = str(exc)
            time.sleep(10)
    raise TimeoutError(f"Endpoint did not become healthy: {last_error}")


def run_load_test(url: str, args):
    total_gpus = args.replicas * 2
    cmd = [
        sys.executable,
        str(ROOT / "load_test_infertutor.py"),
        "--url",
        url,
        "--model",
        args.model,
        "--mode",
        args.mode,
        "--users",
        str(args.users),
        "--duration",
        str(args.duration),
        "--ramp-up",
        str(args.ramp_up),
        "--max-tokens",
        str(args.max_tokens),
        "--label",
        args.label,
        "--total-gpus",
        str(total_gpus),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Deploy and benchmark experimental disaggregated InferTutor")
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--gpu-type", default="H100", choices=["H100", "H200", "A100", "L40S"])
    parser.add_argument("--replicas", type=int, default=2, help="Each replica uses 2 GPUs: 1 prefill + 1 decode.")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-batch-tokens", type=int, default=2048)
    parser.add_argument("--max-seqs", type=int, default=32)
    parser.add_argument("--concurrent-inputs", type=int, default=64)
    parser.add_argument("--mm-max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--mode", choices=["text", "long", "image", "mixed"], default="mixed")
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--duration", type=int, default=45)
    parser.add_argument("--ramp-up", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--url", default=None)
    parser.add_argument("--deploy-only", action="store_true")
    args = parser.parse_args()

    total_gpus = args.replicas * 2
    console.print(Panel(json.dumps(vars(args) | {"total_gpus": total_gpus}, indent=2), title="InferTutor Disagg Experiment"))

    if total_gpus > 8:
        raise SystemExit("This runner caps experiments at 8 GPUs.")

    url = args.url or deploy(patch_modal_app(args))
    wait_for_health(url)
    if not args.deploy_only:
        run_load_test(url, args)


if __name__ == "__main__":
    main()
