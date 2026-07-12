# Optimizing Multimodal LLM Inference with vLLM and Modal: An Ablation Study

Subtitle: What I learned from tuning a Qwen3-VL serving system across scheduler settings, routing, speculative decoding, quantization, compiled mode, and a disaggregated serving prototype.

By Harshavardhana Srinivasan

---

## Introduction

Serving a large language model is not just about getting a model to respond. Under real concurrent traffic, the hard part is balancing latency, throughput, reliability, and GPU cost at the same time.

For this project, I worked on an InferTutor-style multimodal inference workload using **Qwen/Qwen3-VL-4B-Instruct**, **vLLM**, and **Modal-hosted H100 GPUs**. The workload included short text prompts, long-context prompts, image-conditioned prompts, and mixed multimodal traffic.

The goal was simple to state but difficult to optimize:

> Find a serving configuration that can handle concurrent mixed multimodal traffic with low p95 latency, high streamed throughput, zero or near-zero errors, and efficient GPU usage.

The final best configuration was a 4-replica H100 vLLM deployment using:

- **max-seqs=32**
- **max-batch-tokens=2048**
- **prefix caching enabled**
- **chunked prefill enabled**
- **120 mixed users**
- **96 max output tokens**

It achieved:

| Metric | Value |
|---|---:|
| Final score | **7,084,876** |
| TTFT p95 | 638.0 ms |
| ITL p95 | 21.4 ms |
| Throughput | 3231.5 chunks/s |
| Error rate | 0.0% |
| GPUs | 4 H100 replicas |

The most important lesson was not that one advanced serving feature magically solved the problem. The best result came from **scheduler balance**: controlling how prefill-heavy and decode-heavy work shared the same vLLM serving pool.

---

## System Setup

The serving stack was:

| Layer | Choice |
|---|---|
| Model | Qwen/Qwen3-VL-4B-Instruct |
| Serving engine | vLLM OpenAI-compatible server |
| GPU platform | Modal GPU containers |
| Main GPU | H100 |
| Load tester | Async streaming client |
| Main workload | Mixed text, long-context, and image prompts |

The final winning architecture was intentionally simple: a monolithic vLLM pool with four Modal replicas.

**Image placeholder: Final monolithic vLLM serving architecture**

Upload: `medium_assets/architecture_monolithic.png`

In this architecture, each replica runs one vLLM server. Prefill and decode happen inside the same vLLM scheduler. This matters because many of the more complex ideas I tested later did not beat this simpler tuned baseline.

---

## How the Harness Scored Performance

The experiment harness used a throughput-normalized latency efficiency score:

```text
score = goodput_chunks_per_second * users / (ttft_p95_seconds * itl_p95_seconds * total_gpus)
```

where:

```text
goodput_chunks_per_second = aggregate_stream_chunks_per_s * (1 - error_rate)
ttft_p95_seconds = ttft_p95_ms / 1000
itl_p95_seconds = itl_p95_ms / 1000
```

This formula rewards four things at once:

1. **Useful streamed output:** throughput is discounted by request errors.
2. **Concurrent user support:** higher stable user counts improve the score.
3. **Tail latency control:** p95 TTFT and p95 ITL are in the denominator.
4. **GPU efficiency:** using more GPUs only helps if performance improves enough to justify the extra hardware.

This is why a configuration with high raw throughput can still lose if p95 latency explodes or if it uses more GPUs inefficiently.

---

## The Best Configuration

The final best command was:

```bash
python run_infertutor_experiment.py --label mixed-r4-b2048-u120 --gpu-type H100 --replicas 4 --max-seqs 32 --max-batch-tokens 2048 --mode mixed --users 120 --duration 90 --ramp-up 30 --max-tokens 96
```

The important serving settings were:

| Setting | Value |
|---|---:|
| Replicas | 4 |
| GPU type | H100 |
| Mode | mixed |
| Users | 120 |
| Max sequences | 32 |
| Max batched tokens | 2048 |
| Prefix cache | Enabled |
| Chunked prefill | Enabled |

The best run was not the largest batch-token configuration, the highest user count, or the most advanced execution mode. It was the configuration that best balanced active sequence capacity with prefill/decode fairness.

---

## Why max-batch-tokens=2048 Won

One of the most important ablations was the batch-token budget.

| Variant | Result |
|---|---|
| b2048 | Best mixed result |
| b4096 | Prefill-heavy work dominated scheduling windows |
| b1024 | Work was sliced too aggressively and utilization fell |

The b4096 run performed dramatically worse even though it looked like a reasonable larger-batch setting. The issue was that large prefill-heavy image and long-context requests could occupy too much of the scheduling window, increasing time to first token.

The b1024 run also regressed. It limited prefill pressure, but sliced work too aggressively and reduced the ability of the GPUs to stay productive.

The b2048 setting was the best middle ground.

---

## Prefix Caching and Chunked Prefill Were Not Optional

Two vLLM features were especially important for the final configuration:

| Feature | Observation |
|---|---|
| Prefix caching | Disabling it sharply reduced the score |
| Chunked prefill | Disabling it increased latency and hurt scheduler balance |

The result was clear: for this mixed workload, both prefix caching and chunked prefill should remain enabled.

Chunked prefill was particularly important because mixed multimodal workloads contain requests with very different prompt costs. Without chunked prefill, long/image prompts can interfere more severely with latency-sensitive decode work.

---

## User-Density Testing: 120 Users Was the Knee

The best 4-replica mixed configuration was stable at 120 users. When I pushed to 130 and 140 users, the system began to collapse from queueing pressure.

| Users | Outcome |
|---:|---|
| 120 | Best stable mixed result |
| 130 | Queueing increased and errors appeared |
| 140 | Severe latency collapse |

This was a useful reminder that more users do not automatically mean a better score. If p95 TTFT and ITL rise faster than throughput, the score collapses.

---

## Sequence Concurrency: seq32 Was the Balance Point

I also tested different active sequence windows:

| max-seqs | Result |
|---:|---|
| 16 | Too restrictive; queues grew |
| 32 | Best observed setting |
| 48 | Too many active sequences; contention increased |

The seq16 run limited concurrency too much. The seq48 run allowed too much active work and caused contention. The seq32 setting provided the best observed balance between keeping replicas busy and keeping p95 latency under control.

---

## What Happened When I Tried More Advanced Methods

After establishing the best monolithic configuration, I tested several more novel directions.

| Method | Outcome | Main lesson |
|---|---|---|
| N-gram speculative decoding | Rejected | Overhead outweighed benefit for this mixed workload |
| Online FP8 weight quantization | Rejected | Did not improve latency-throughput balance and introduced errors |
| Modal concurrency gate | Rejected | External ingress limiting hurt effective throughput |
| Compiled mode | Rejected for final score | Improved ITL but worsened TTFT and overall score |
| Request-class routing | Rejected for final score | Better with 1+3 than 2+2, but still below monolithic |
| Disaggregated serving | Attempted but not scored | Blocked by vLLM/LMCache compatibility |

This is where the project became most interesting. Several techniques that sounded promising did not transfer cleanly to this specific mixed multimodal workload.

---

## Request-Class Routing: 2+2 vs 1+3

One architectural experiment was to split traffic by request type.

The first routed design used:

- 2 replicas for text requests
- 2 replicas for long/image requests

**Image placeholder: Request-class routing 2+2 architecture**

Upload: `medium_assets/architecture_routed.png`

This did not work well. The heavy endpoint became constrained because long-context and image requests were more expensive, but only had two replicas.

Then I tested an asymmetric routing design:

- 1 replica for text requests
- 3 replicas for long/image requests

**Image placeholder: Asymmetric request-class routing 1+3 architecture**

Upload: `medium_assets/architecture_routed_asymmetric.png`

The 1+3 design improved over 2+2:

| Routing design | Score | Error rate | Observation |
|---|---:|---:|---|
| 2+2 split | 235,765 | 0.1% | Heavy pool constrained |
| 1+3 split | 621,303 | 0.0% | Better allocation, still below monolithic |
| Monolithic best | 7,084,876 | 0.0% | Best overall |

The lesson was subtle: request routing was not useless, but naive separation was not enough. Replica allocation must match workload cost. Even after improving from 2+2 to 1+3, the routed system still lost to the single tuned vLLM pool.

---

## Disaggregated Prefill/Decode Prototype

The most ambitious architecture I attempted was disaggregated prefill/decode serving.

The idea was to separate the two major transformer inference phases:

- **Prefill:** process the prompt and build KV cache
- **Decode:** generate output tokens incrementally

In principle, this can reduce interference between long prompt processing and latency-sensitive token generation.

**Image placeholder: Attempted disaggregated prefill/decode architecture**

Upload: `medium_assets/architecture_disaggregated.png`

For this project, I did not build a new inference engine from scratch. The prototype used:

- vLLM as the intended prefiller and decoder engine
- LMCache/NIXL-style KV transfer as the intended KV-cache movement mechanism
- project-specific Modal and FastAPI orchestration to launch roles, proxy requests, and expose health checks

The prototype did not become score-producing because the available LMCache integration was incompatible with the installed vLLM version:

```text
lmcache.integration.vllm.vllm_v1_adapter imported cdiv from vllm.utils,
but vllm==0.21.0 did not expose cdiv.
```

So the decision was to document the prototype rather than force it into the final benchmark. The failure was not a normal performance regression. It was a dependency-alignment problem.

This still taught an important systems lesson: advanced serving architectures create integration risk, not just performance risk.

---

## Score Comparison

**Image placeholder: Score comparison across key runs**

Upload: `medium_assets/score_comparison.png`

The best monolithic mixed configuration was the clear winner in the comparable 4-GPU mixed setting.

| Configuration | Score |
|---|---:|
| Final best mixed scheduler | **7,084,876** |
| Compiled mixed execution | 4,097,013 |
| Text-only single-replica knee | 3,380,964 |
| Mixed two-replica baseline | 2,687,395 |
| Request routing 1+3 | 621,303 |
| Request routing 2+2 | 235,765 |

The compiled run is worth calling out. It improved inter-token latency, but worsened TTFT and reduced throughput enough that it did not beat the eager b2048 configuration.

---

## Latency and Throughput Tradeoff

**Image placeholder: Latency and throughput tradeoff across mixed runs**

Upload: `medium_assets/latency_throughput.png`

The winning run sat in the low-TTFT, high-throughput region. Most failed experiments moved in the wrong direction: p95 TTFT increased, throughput dropped, or both.

This is why p95 latency was so central. Average latency can hide bad tail behavior, but p95 latency reveals when the system begins to saturate.

---

## Final Lessons

Here are the main things I took away from the study.

### 1. Mixed multimodal serving is scheduler-sensitive

The biggest gains came from balancing prefill and decode pressure. The best configuration was not the most complicated architecture; it was the best scheduler balance inside a monolithic vLLM deployment.

### 2. Larger batch settings can hurt tail latency

Increasing max-batch-tokens to 4096 looked plausible, but it allowed prefill-heavy requests to dominate too much of the scheduling window.

### 3. Advanced techniques are workload-dependent

Speculative decoding, quantization, compiled mode, and request routing can all be useful in the right setting. In this mixed multimodal workload, they did not beat the tuned eager vLLM baseline.

### 4. Routing needs cost-aware allocation

The 1+3 routing design was better than 2+2 because heavy requests needed more capacity. But routing still did not beat the monolithic pool.

### 5. Disaggregated serving needs dependency alignment

The disaggregated prototype was conceptually interesting but blocked by LMCache/vLLM compatibility. Before benchmarking such systems, the version matrix needs to be validated with a single-request smoke test.

---

## Conclusion

The final recommended configuration was a 4-replica H100 mixed vLLM endpoint with max-seqs=32, max-batch-tokens=2048, prefix caching enabled, and chunked prefill enabled.

It achieved the best observed balance of:

- time to first token,
- inter-token latency,
- streamed throughput,
- reliability,
- concurrency,
- GPU efficiency.

The most important result was not just the final score. It was the engineering path: run controlled ablations, keep p95 latency visible, treat failed experiments as evidence, and avoid assuming that advanced serving features automatically improve every workload.

For this workload, stable prefill/decode interleaving inside a tuned monolithic vLLM scheduler was the central performance lever.

---

## Repository and Full Report

GitHub repository:

https://github.com/asharsha30-1996/LLM-INFERENCINGab-ABLATION-STUDY

Full artifacts included in the repository:

- `EXPERIMENT_REPORT.pdf`
- `DISAGGREGATED_SERVING_DETAILS.pdf`
- `ARCHITECTURE_NOVEL_METHODS.pdf`
- report diagrams under `report_assets/`
- Modal/vLLM experiment code under `starter_code/`

---

## Medium Image Upload Checklist

Upload these images manually in the article where the placeholders appear:

1. `medium_assets/architecture_monolithic.png`
2. `medium_assets/architecture_routed.png`
3. `medium_assets/architecture_routed_asymmetric.png`
4. `medium_assets/architecture_disaggregated.png`
5. `medium_assets/score_comparison.png`
6. `medium_assets/latency_throughput.png`

PNG versions are available in medium_assets/ and are ready for Medium upload.