"""
Experimental InferTutor disaggregated-prefill Modal app.

Each Modal container uses 2 GPUs:
- GPU 0 runs the vLLM prefiller.
- GPU 1 runs the vLLM decoder.
- A local FastAPI proxy receives OpenAI-compatible requests, sends a 1-token
  prefill request to the prefiller, then streams the full decode response.

This follows the vLLM experimental LMCache/NIXL disaggregated-prefill pattern.
It is intentionally separate from the stable monolithic app.
"""

import os
import subprocess
import textwrap
import time
from pathlib import Path

import modal


vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.21.0",
        "qwen-vl-utils==0.0.14",
        "lmcache",
        "nixl",
        "fastapi",
        "uvicorn",
        "httpx",
        "numpy",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

app = modal.App("infertutor-disagg-pd")

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)


# These constants are patched by run_disagg_infertutor_experiment.py before deploy.
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-4B-Instruct")
GPU_TYPE = os.environ.get("GPU_TYPE", "H100")
DTYPE = os.environ.get("DTYPE", "bfloat16")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
MAX_NUM_BATCHED_TOKENS = int(os.environ.get("MAX_NUM_BATCHED_TOKENS", "2048"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "32"))
CONCURRENT_INPUTS = int(os.environ.get("CONCURRENT_INPUTS", "64"))
MIN_CONTAINERS = int(os.environ.get("MIN_CONTAINERS", "2"))
MAX_CONTAINERS = int(os.environ.get("MAX_CONTAINERS", "2"))
MM_MAX_PIXELS = int(os.environ.get("MM_MAX_PIXELS", str(512 * 28 * 28)))

MINUTES = 60
PROXY_PORT = 8000
PREFILL_PORT = 8100
DECODE_PORT = 8200


PROXY_SOURCE = r'''
import argparse
import json
import os
import time
from contextlib import asynccontextmanager

import httpx
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


class StatsCalculator:
    def __init__(self):
        self._stats = []
        self._last_log_time = time.time()

    def add(self, value):
        self._stats.append(value)
        if time.time() - self._last_log_time > 10 and self._stats:
            arr = np.array(self._stats)
            print(
                "Prefill proxy stats:",
                f"n={len(self._stats)}",
                f"avg_ms={np.mean(arr) * 1000:.1f}",
                f"p50_ms={np.percentile(arr, 50) * 1000:.1f}",
                f"p99_ms={np.percentile(arr, 99) * 1000:.1f}",
                flush=True,
            )
            self._last_log_time = time.time()


stats_calculator = StatsCalculator()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--prefiller-port", type=int, default=8100)
    parser.add_argument("--decoder-port", type=int, default=8200)
    return parser.parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prefill_client = httpx.AsyncClient(
        timeout=None,
        base_url=f"http://127.0.0.1:{global_args.prefiller_port}/v1",
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
    )
    app.state.decode_client = httpx.AsyncClient(
        timeout=None,
        base_url=f"http://127.0.0.1:{global_args.decoder_port}/v1",
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
    )
    yield
    await app.state.prefill_client.aclose()
    await app.state.decode_client.aclose()


app = FastAPI(lifespan=lifespan)


def tail_file(path, max_chars=2000):
    try:
        with open(path, "r", errors="replace") as handle:
            data = handle.read()
        return data[-max_chars:]
    except Exception as exc:
        return f"could not read {path}: {exc}"


@app.get("/health")
async def health():
    details = {
        "prefill": None,
        "decode": None,
        "prefill_log_tail": tail_file("/tmp/prefiller.log", max_chars=3000),
        "decode_log_tail": tail_file("/tmp/decoder.log", max_chars=3000),
    }

    async with httpx.AsyncClient(timeout=5) as client:
        try:
            prefill = await client.get(f"http://127.0.0.1:{global_args.prefiller_port}/health")
            details["prefill"] = {"status": prefill.status_code, "body": prefill.text[:500]}
        except Exception as exc:
            details["prefill"] = {"error": str(exc)}

        try:
            decode = await client.get(f"http://127.0.0.1:{global_args.decoder_port}/health")
            details["decode"] = {"status": decode.status_code, "body": decode.text[:500]}
        except Exception as exc:
            details["decode"] = {"error": str(exc)}

    prefill_ok = details["prefill"] and details["prefill"].get("status") == 200
    decode_ok = details["decode"] and details["decode"].get("status") == 200
    if prefill_ok and decode_ok:
        return JSONResponse(details, status_code=200)

    try:
        summary = {
            "prefill": details["prefill"],
            "decode": details["decode"],
            "prefill_log_tail": details["prefill_log_tail"][-1200:],
            "decode_log_tail": details["decode_log_tail"][-1200:],
        }
        print("Disagg health failure:", json.dumps(summary), flush=True)
    except Exception:
        print("Disagg health exception: could not serialize details", flush=True)
    return JSONResponse(details, status_code=503)


async def send_prefill(client, endpoint, req_data):
    req_data = req_data.copy()
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}"}
    response = await client.post(endpoint, json=req_data, headers=headers)
    response.raise_for_status()
    await response.aread()


async def stream_decode(client, endpoint, req_data):
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}"}
    async with client.stream("POST", endpoint, json=req_data, headers=headers) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


async def handle_openai_request(request, endpoint):
    req_data = await request.json()
    start = time.time()
    await send_prefill(request.app.state.prefill_client, endpoint, req_data)
    stats_calculator.add(time.time() - start)

    async def generate():
        async for chunk in stream_decode(request.app.state.decode_client, endpoint, req_data):
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/v1/completions")
async def completions(request: Request):
    return await handle_openai_request(request, "/completions")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await handle_openai_request(request, "/chat/completions")


if __name__ == "__main__":
    global_args = parse_args()
    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
'''


def write_lmcache_configs():
    config_dir = Path("/tmp/disagg_configs")
    config_dir.mkdir(parents=True, exist_ok=True)
    prefiller_config = config_dir / "lmcache-prefiller-config.yaml"
    decoder_config = config_dir / "lmcache-decoder-config.yaml"
    prefiller_config.write_text(
        textwrap.dedent(
            """
            local_cpu: False
            max_local_cpu_size: 0
            remote_serde: NULL
            enable_nixl: True
            nixl_role: "sender"
            nixl_peer_host: "localhost"
            nixl_peer_port: 55555
            nixl_buffer_size: 1073741824
            nixl_buffer_device: "cuda"
            nixl_enable_gc: True
            """
        ).strip()
        + "\n"
    )
    decoder_config.write_text(
        textwrap.dedent(
            """
            local_cpu: False
            max_local_cpu_size: 0
            remote_serde: NULL
            enable_nixl: True
            nixl_role: "receiver"
            nixl_peer_host: "localhost"
            nixl_peer_port: 55555
            nixl_buffer_size: 1073741824
            nixl_buffer_device: "cuda"
            nixl_enable_gc: True
            """
        ).strip()
        + "\n"
    )
    return prefiller_config, decoder_config


def base_vllm_cmd(port: int):
    return [
        "vllm",
        "serve",
        MODEL_NAME,
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tensor-parallel-size",
        "1",
        "--dtype",
        DTYPE,
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-batched-tokens",
        str(MAX_NUM_BATCHED_TOKENS),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--gpu-memory-utilization",
        "0.90",
        "--uvicorn-log-level=warning",
        "--enforce-eager",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--limit-mm-per-prompt",
        '{"image": 1, "video": 0}',
        "--mm-processor-kwargs",
        f'{{"min_pixels": 784, "max_pixels": {MM_MAX_PIXELS}, "fps": 1}}',
    ]


@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:2",
    scaledown_window=60,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    timeout=20 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
@modal.concurrent(max_inputs=CONCURRENT_INPUTS)
@modal.web_server(port=PROXY_PORT, startup_timeout=20 * MINUTES)
def serve():
    prefiller_config, decoder_config = write_lmcache_configs()
    proxy_path = Path("/tmp/disagg_proxy_server.py")
    proxy_path.write_text(PROXY_SOURCE)

    env_base = os.environ.copy()
    env_base["PYTHONHASHSEED"] = "123"
    env_base["PYTHONUNBUFFERED"] = "1"
    env_base["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env_base["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    env_base["UCX_TLS"] = "cuda_ipc,cuda_copy,tcp"

    prefill_env = env_base.copy()
    prefill_env["CUDA_VISIBLE_DEVICES"] = "0"
    prefill_env["LMCACHE_CONFIG_FILE"] = str(prefiller_config)
    prefill_cmd = base_vllm_cmd(PREFILL_PORT) + [
        "--kv-transfer-config",
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config":{"discard_partial_chunks":false,"lmcache_rpc_port":"producer1"}}',
    ]

    decode_env = env_base.copy()
    decode_env["CUDA_VISIBLE_DEVICES"] = "1"
    decode_env["LMCACHE_CONFIG_FILE"] = str(decoder_config)
    decode_cmd = base_vllm_cmd(DECODE_PORT) + [
        "--kv-transfer-config",
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config":{"discard_partial_chunks":false,"lmcache_rpc_port":"consumer1"}}',
    ]

    print("Starting disaggregated prefiller:", " ".join(prefill_cmd), flush=True)
    prefiller_log = open("/tmp/prefiller.log", "w", buffering=1)
    subprocess.Popen(prefill_cmd, env=prefill_env, stdout=prefiller_log, stderr=subprocess.STDOUT)
    print("Starting disaggregated decoder:", " ".join(decode_cmd), flush=True)
    decoder_log = open("/tmp/decoder.log", "w", buffering=1)
    subprocess.Popen(decode_cmd, env=decode_env, stdout=decoder_log, stderr=subprocess.STDOUT)

    # Give both vLLM processes a head start before exposing the proxy.
    time.sleep(30)
    proxy_cmd = [
        "python",
        str(proxy_path),
        "--host",
        "0.0.0.0",
        "--port",
        str(PROXY_PORT),
        "--prefiller-port",
        str(PREFILL_PORT),
        "--decoder-port",
        str(DECODE_PORT),
    ]
    print("Starting disaggregated proxy:", " ".join(proxy_cmd), flush=True)
    subprocess.Popen(proxy_cmd)
