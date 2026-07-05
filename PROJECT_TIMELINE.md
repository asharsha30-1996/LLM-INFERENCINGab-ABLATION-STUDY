# InferTutor Arena Project Timeline

This is our working plan for the LLM inference capstone. The goal is to run disciplined ablation studies for `Qwen/Qwen3-VL-4B-Instruct` using vLLM on Modal-hosted GPUs, then submit a clean engineering report with reproducible commands and benchmark JSONs.

## North Star

Optimize the official score:

```text
Score =
  goodput_tokens_per_second
  * sustained_users
  * quality_pass_rate
  * (1 - error_rate)
  / (p95_TTFT_seconds * p95_ITL_seconds * total_GPU_count)
```

For the starter harness, throughput is measured as streamed content chunks per second. We will optimize the same way the final evaluator likely rewards: high clean throughput, low p95 TTFT, low p95 ITL, low errors, and no wasteful GPU scaling.

## Constraints

- Model: `Qwen/Qwen3-VL-4B-Instruct`
- Serving engine: vLLM OpenAI-compatible server
- Hosting: Modal
- Main GPU target: H100
- Main track: `--mode mixed`
- Text speed track: `--mode text`
- Recommended budget: up to 4 H100 GPUs
- Official prompts must stay unchanged for submitted runs.
- Scoring script and workload mode must stay unchanged for submitted runs.

## Phase 0: Repo and Environment Readiness

Target outcome: local machine can deploy a Modal vLLM endpoint and run the load tester.

Checklist:

- Confirm Python 3.11+ is available.
- Install starter dependencies from `starter_code/requirements.txt`.
- Authenticate Modal.
- Create the Modal Hugging Face secret named `huggingface`.
- Confirm Git is installed and repo is clean.
- Keep the two reference reports nearby:
  - `C:\Users\HarshavardhanaSriniv\Desktop\Harsha-Projects\References\capstone_report_v4_inference_newest.pdf`
  - `C:\Users\HarshavardhanaSriniv\Desktop\Harsha-Projects\References\InferTutor Arena - Manoj.pdf`

Commands:

```bash
cd starter_code
pip install -r requirements.txt
modal token info
modal secret list
```

If Modal or Hugging Face auth is missing:

```bash
modal token new
modal secret create huggingface HF_TOKEN=<YOUR_HF_TOKEN>
```

## Phase 1: Smoke Test

Target outcome: prove infrastructure works with a tiny text run before spending GPU time.

Command:

```bash
python run_infertutor_experiment.py \
  --label smoke \
  --gpu-type H100 \
  --replicas 1 \
  --mode text \
  --users 5 \
  --duration 30 \
  --ramp-up 5 \
  --max-tokens 64
```

Pass criteria:

- Endpoint becomes healthy.
- Result JSON is saved under `starter_code/results_infertutor/`.
- Error rate is 0%.
- TTFT and ITL are nonzero and plausible.

## Phase 2: Baseline Runs

Target outcome: establish reference points before changing knobs.

### Single-GPU Text Baseline

```bash
python run_infertutor_experiment.py \
  --label baseline-text-r1 \
  --gpu-type H100 \
  --replicas 1 \
  --max-seqs 32 \
  --max-batch-tokens 4096 \
  --mode text \
  --users 50 \
  --duration 60 \
  --ramp-up 15 \
  --max-tokens 96
```

### Multimodal Product Baseline

```bash
python run_infertutor_experiment.py \
  --label baseline-mixed-r2 \
  --gpu-type H100 \
  --replicas 2 \
  --max-seqs 32 \
  --max-batch-tokens 4096 \
  --mode mixed \
  --users 100 \
  --duration 90 \
  --ramp-up 25 \
  --max-tokens 96
```

Score after runs:

```bash
python score_infertutor.py results_infertutor
```

## Phase 3: Find the Concurrency Bend

Target outcome: identify where p95 TTFT, p95 ITL, or errors start bending upward.

Keep serving config fixed and sweep users:

| Label | Mode | Replicas | Users | Purpose |
|---|---|---:|---:|---|
| mixed-r2-u60 | mixed | 2 | 60 | Low-load reference |
| mixed-r2-u100 | mixed | 2 | 100 | Assignment baseline region |
| mixed-r2-u140 | mixed | 2 | 140 | Stress tail latency |
| mixed-r2-u180 | mixed | 2 | 180 | Find error/latency cliff |

Hold these fixed unless a run fails badly:

```text
--max-seqs 32
--max-batch-tokens 4096
--prefix-cache
--chunked-prefill
--fast-boot
--max-tokens 96
```

Decision rule:

- If TTFT p95 rises sharply while ITL remains acceptable, suspect prefill queueing.
- If ITL p95 rises sharply, suspect oversized decode batches or too much per-container concurrency.
- If errors appear, reduce users, reduce `--concurrent-inputs`, or add replicas.

## Reference-Inspired Strategy

The two reference submissions are useful because they show two different levels of maturity:

- Manoj's report is a strong baseline-style journey: smoke test, single-GPU saturation, prefix cache, chunked prefill, replicas, compiled mode, mixed-mode ceiling, image pixel budget, and final report polish.
- Oluwaseyi's report is a much deeper systems study: compiled-mode density, ramp-up calibration, H100/B200 comparisons, FP8 KV cache, MXFP8 KV cache, NVFP4 attempt, tensor parallelism, speculative decoding attempts, and remaining unexplored directions.

Key lessons we should reuse:

- Compiled mode is the biggest text-track lever, but it is risky or ineffective for mixed multimodal traffic.
- Mixed mode likes smaller batch-token budgets than text mode. One reference found `--max-batch-tokens 2048` useful because it lets chunked prefill interleave long/image prefills more aggressively.
- `--max-seqs 32` is a strong default. Higher values can improve saturation but often hurt ITL or TTFT.
- Prefix caching is not guaranteed to help. One reference found it hurt for short prompts; another found it helped at higher compiled text concurrency. We should measure it in our exact regime.
- Ramp-up matters. Longer ramp-up can reduce cold-start and compile-related errors in larger compiled runs.
- FP8 KV cache on H100 is probably not a winning final setting because prior results showed ITL rising from dequantization overhead. It is still worth one controlled ablation if we can frame it as a negative result.
- Image pixel budget is context-dependent. Lower pixels can help overloaded runs by reducing image prefill cost, but can hurt healthy runs through scheduler fragmentation or quality loss.

Our project should not merely replay the references. We will use their best known settings as launchpads, then reserve a small budget for targeted novelty.

## Phase 4: Core Ablations

Target outcome: test one serving hypothesis at a time.

Use the best user count from Phase 3 as the stable load point.

| Ablation | Config A | Config B | Hypothesis |
|---|---|---|---|
| Prefix caching | `--prefix-cache` | `--no-prefix-cache` | Shared system prompt should benefit from KV reuse. |
| Chunked prefill | `--chunked-prefill` | `--no-chunked-prefill` | Mixed long/image traffic should benefit from chunked prefill. |
| Batch tokens | `--max-batch-tokens 2048` | `--max-batch-tokens 4096` / `8192` | Smaller batches may lower TTFT; larger batches may improve throughput. |
| Max sequences | `--max-seqs 16` | `--max-seqs 32` / `64` | Wider batches may improve throughput but hurt ITL. |
| Modal concurrency | `--concurrent-inputs 32` | `--concurrent-inputs 64` / `96` | Too much web concurrency can overload vLLM queues. |
| Image pixel budget | default `401408` | lower value such as `200704` | Lower image preprocessing cost may help mixed traffic if quality remains acceptable. |

Reference-inspired mixed candidate:

```bash
python run_infertutor_experiment.py \
  --label mixed-r4-b2048-u120 \
  --gpu-type H100 \
  --replicas 4 \
  --max-seqs 32 \
  --max-batch-tokens 2048 \
  --mode mixed \
  --users 120 \
  --duration 90 \
  --ramp-up 30 \
  --max-tokens 96
```

Scoring cadence:

```bash
python score_infertutor.py results_infertutor
```

After each run, record:

- Command
- Result JSON filename
- TTFT p95
- ITL p95
- Throughput
- Error rate
- Score
- Keep/reject decision
- One-sentence interpretation

## Phase 5: Scale-Out Runs

Target outcome: determine whether more replicas improve score enough to justify GPU count.

Recommended mixed-mode scale sweep:

| Label | Mode | Replicas | Users | Notes |
|---|---|---:|---:|---|
| mixed-r1-best | mixed | 1 | best stable users for 1 GPU | Efficiency baseline |
| mixed-r2-best | mixed | 2 | best stable users for 2 GPUs | Main contender |
| mixed-r4-best | mixed | 4 | higher stable users | Beat internal 4-replica reference |

Internal mixed reference to beat:

| Config | Users | GPUs | Errors | TTFT p95 | ITL p95 | Throughput |
|---|---:|---:|---:|---:|---:|---:|
| eager, 4 replicas, seq32/b4096 | 120 | 4 | 0.0% | 897.6 ms | 38.1 ms | 2,756 chunks/s |
| eager, 2 replicas, seq32/b4096 | 100 | 2 | 0.0% | 1,168.9 ms | 28.7 ms | 2,243 chunks/s |

Decision rule:

- Keep the replica count only if the score improves after GPU penalty.
- Prefer 0% errors over slightly higher raw throughput.
- Watch p95 metrics, not just averages.

## Phase 6: Text Track Optional

Target outcome: decide whether to submit or report a separate text-speed result.

Compiled mode is likely useful for text-only traffic:

```bash
python run_infertutor_experiment.py \
  --label text-compiled-r4 \
  --gpu-type H100 \
  --replicas 4 \
  --no-fast-boot \
  --max-seqs 32 \
  --max-batch-tokens 4096 \
  --mode text \
  --users 400 \
  --duration 90 \
  --ramp-up 40 \
  --max-tokens 96
```

Internal text reference to beat:

| Config | Users | GPUs | Errors | TTFT p95 | ITL p95 | Throughput |
|---|---:|---:|---:|---:|---:|---:|
| compiled, 4 replicas, seq32/b4096 | 400 | 4 | 0.0% | 1,942.5 ms | 16.2 ms | 11,064 chunks/s |
| compiled, 1 replica, seq32/b4096 | 100 | 1 | 0.0% | 1,266.8 ms | 10.8 ms | 3,570 chunks/s |

We should not use compiled mode for mixed traffic unless an explicit ablation proves it is stable and worthwhile.

## Phase 6B: Novel Experiment Lane

Target outcome: add one or two experiments that are intellectually distinct from the reference submissions while staying measurable and reproducible.

The starter runner now supports a generic pass-through flag:

```bash
--vllm-arg <one-token>
```

Repeat it for each raw `vllm serve` argument token. This lets us test vLLM features that are not first-class starter flags.

### Novel Option A: N-Gram Speculation in Eager Text Mode

Why this is interesting:

- Prior reference work attempted speculative decoding with a draft model and hit VL/static-shape issues.
- vLLM's n-gram speculation does not require a second draft model; it proposes tokens by matching prompt n-grams.
- It may work better as a text-mode side study than as a mixed-mode final setting.

Hypothesis:

- On repeated tutor-style text prompts, n-gram speculation may lower ITL without requiring extra GPUs.
- It may also increase overhead or fail to help because generated answers are not highly copied from the prompt.

Candidate command:

```bash
python run_infertutor_experiment.py \
  --label text-ngram-eager-r1-u75 \
  --gpu-type H100 \
  --replicas 1 \
  --fast-boot \
  --max-seqs 32 \
  --max-batch-tokens 4096 \
  --mode text \
  --users 75 \
  --duration 90 \
  --ramp-up 30 \
  --max-tokens 96 \
  --vllm-arg=--speculative-config \
  --vllm-arg '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4}'
```

Decision rule:

- Keep only if ITL p95 improves without TTFT or error-rate damage.
- If startup fails or hangs, record it as compatibility/operational evidence and move on.

### Novel Option B: Online FP8 Weight Quantization

Why this is interesting:

- Prior references tested FP8 KV cache and NVFP4-style attempts, but not necessarily online FP8 linear-weight quantization on our exact starter path.
- vLLM's current docs describe online quantization for BF16/FP16 checkpoints at load time using schemes such as `fp8_per_tensor` and `fp8_per_block`.

Hypothesis:

- Weight quantization may reduce memory bandwidth and improve throughput, but the 4B model already fits comfortably on H100, so compute/quant overhead may erase gains.
- This is likely a side-study, not our first final-track bet.

Candidate command:

```bash
python run_infertutor_experiment.py \
  --label text-fp8-weight-r1-u75 \
  --gpu-type H100 \
  --replicas 1 \
  --max-seqs 32 \
  --max-batch-tokens 4096 \
  --mode text \
  --users 75 \
  --duration 90 \
  --ramp-up 30 \
  --max-tokens 96 \
  --vllm-arg=--quantization \
  --vllm-arg fp8_per_tensor
```

Decision rule:

- Compare against the same text run without quantization.
- Watch startup time, ITL p95, throughput, and error rate.
- If it does not beat BF16, it still becomes a good negative-result paragraph.

### Novel Option C: Mixed-Mode Request-Class Specialization

Why this is interesting:

- Both references identify mixed traffic as harder because image/long prefills interfere with short text decode.
- The clean systems idea is to stop treating mixed traffic as one undifferentiated pool.

Implementation path:

- Deploy two endpoints with different configs:
  - Text-leaning endpoint: compiled or high-throughput config for text-only traffic.
  - Vision/long endpoint: eager, chunked-prefill, lower `max-batch-tokens`, image-safe config.
- Build a small router load tester that sends official text prompts to one endpoint and image/long prompts to the other.

Caveat:

- This may not be allowed for final submission if the evaluator expects one endpoint and one official mixed-mode command.
- It is still a strong engineering extension and report section if we clearly label it as an exploratory architecture experiment.

Decision rule:

- Do this only after our normal mixed final run is secure.
- Treat it as a product-systems experiment, not the official final unless rules allow multiple endpoints.

### Novel Option D: Controlled FP8 KV Negative Result

Why this is interesting:

- Prior work suggests FP8 KV hurts H100 ITL. We can reproduce that on a smaller, cheaper scale to show we understand hardware-dependent quantization.

Candidate command:

```bash
python run_infertutor_experiment.py \
  --label text-fp8-kv-r1-u75 \
  --gpu-type H100 \
  --replicas 1 \
  --max-seqs 32 \
  --max-batch-tokens 4096 \
  --mode text \
  --users 75 \
  --duration 90 \
  --ramp-up 30 \
  --max-tokens 96 \
  --vllm-arg=--kv-cache-dtype \
  --vllm-arg fp8
```

Decision rule:

- If ITL rises, we have a clean explanation: KV memory savings are outweighed by per-step dequantization overhead on H100.
- If it unexpectedly helps, run a second confirmation at higher users.

## Phase 7: Final Run and Submission Package

Target outcome: one clean final benchmark plus an engineering story.

Final run requirements:

- Use official prompt set.
- Use chosen track mode.
- Run long enough to be credible: usually 90 seconds or more.
- Keep error rate as close to 0% as possible.
- Save exact command and result JSON.
- Stop Modal apps after collecting results.

Cleanup:

```bash
modal app list
modal app stop <APP_ID_OR_NAME>
```

Submission contents:

- Final benchmark JSON.
- Exact final command.
- One-page engineering report.
- Table of at least five experiments.
- Explanation of best configuration.
- One surprising failure or tradeoff.

## Report Skeleton

```text
Title: InferTutor Arena Inference Engineering Report

Final configuration:
- Mode:
- Model:
- GPU type/count:
- Replicas:
- max_num_seqs:
- max_num_batched_tokens:
- prefix caching:
- chunked prefill:
- eager/compiled:
- concurrent inputs:
- image pixel budget:
- users/duration/ramp-up:

Final metrics:
- Score:
- TTFT p95:
- ITL p95:
- Throughput:
- Error rate:
- Requests/s:

Experiment table:
- Baseline
- User sweep
- Prefix cache ablation
- Chunked prefill ablation
- Batch/max-seqs ablation
- Scale-out ablation
- Final run

Interpretation:
- Main bottleneck observed:
- Biggest improvement:
- Failed or surprising optimization:
- What we would try next:
```

## References Folder Note

The reference submissions live outside this repo at `C:\Users\HarshavardhanaSriniv\Desktop\Harsha-Projects\References`. We extracted local text copies under `References\extracted_text` for analysis.

We should use them for:

- Report structure
- Experiment table style
- Final command formatting
- How they explain failed ablations
- Any hidden grading expectations

We should not copy their path blindly. Our differentiator should be one measured novelty lane plus a crisp explanation of why it did or did not improve the actual score.
