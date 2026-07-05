# Architecture and Novel Methods Summary

Author: Harshavardhana Srinivasan

Project: InferTutor Arena LLM inference ablation study.

## Purpose

This document summarizes the architecture changes and novel methods explored during the InferTutor Arena ablation study. It complements the main experiment report by focusing on design intent: what we changed, why we changed it, and what the result taught us.

## Baseline Architecture

The final winning configuration used a monolithic vLLM serving pool:

![Final monolithic vLLM serving architecture](report_assets/architecture_monolithic.svg)

In this design, each Modal replica runs one vLLM OpenAI-compatible server. Prefill and decode are scheduled together inside the same vLLM engine. This design was simple, robust, and ultimately the best-performing configuration under the fixed 4-GPU mixed workload.

Final best configuration:

```text
replicas = 4
max_seqs = 32
max_batch_tokens = 2048
prefix_cache = enabled
chunked_prefill = enabled
users = 120
mode = mixed
```

Final best score:

```text
7,084,876
```

## Scheduler-Level Tuning

The most important architectural change was not a new serving topology. It was scheduler balancing inside the monolithic vLLM pool.

The core observation was that mixed multimodal traffic creates interference between:

- short text requests,
- long-context requests,
- image-conditioned requests,
- prefill-heavy work,
- latency-sensitive decode work.

The best result came from setting max-batch-tokens to 2048 and max-seqs to 32. This gave the scheduler enough active sequence capacity without allowing large prefills to dominate the scheduling window.

## Batch-Token Ablations

| Variant | Result |
|---|---|
| b2048 | Best mixed result |
| b4096 | Too much prefill work accumulated in each batch, increasing TTFT |
| b1024 | Work was sliced too aggressively, reducing utilization and score |

The result showed that batch-token budget is a major control knob for multimodal inference workloads.

## Prefix Cache and Chunked Prefill

Two core vLLM optimizations were tested through ablation:

| Feature | Observation |
|---|---|
| Prefix caching | Disabling it sharply reduced the score |
| Chunked prefill | Disabling it increased latency and reduced scheduling balance |

Both features remained enabled in the final configuration.

## Request-Class Routing

A more architectural experiment split traffic by request class:

![Request-class routing architecture](report_assets/architecture_routed.svg)

The first routed prototype used a 2+2 GPU split:

- 2 replicas for short text traffic,
- 2 replicas for heavy long/image traffic.

This did not beat the monolithic pool because the heavy endpoint became constrained. The heavy pool carried the most expensive requests with only two replicas.

## Asymmetric Request-Class Routing

A second routed prototype shifted capacity toward heavy requests:

![Asymmetric request-class routing architecture](report_assets/architecture_routed_asymmetric.svg)

The 1+3 split used:

- 1 replica for short text traffic,
- 3 replicas for long/image traffic.

This improved over the 2+2 routing design. The score increased from 235,765 to 621,303, and the error rate dropped to 0.0%. However, it still remained below the monolithic 4-replica mixed endpoint.

The result suggests that routing can help only if the replica allocation matches workload cost. Naive separation is not enough.

## N-Gram Speculative Decoding

N-gram speculative decoding was tested as a lightweight speculative execution method. The hypothesis was that repeated prompt structure might allow the server to draft tokens more efficiently.

The result was negative for this workload. The mixed benchmark used relatively short outputs and multimodal prompt diversity, so speculative overhead did not translate into a score improvement.

This result does not reject speculative decoding in general. It shows that for this specific mixed workload and model configuration, N-gram speculation was not beneficial.

## Online FP8 Weight Quantization

Online FP8 weight quantization was tested to see whether reduced weight precision could improve throughput or memory behavior.

The result was also negative. The online quantized run introduced overhead and errors, and it did not improve the latency-throughput balance. A future quantization study should use pre-quantized model artifacts rather than relying on online quantization during serving.

## Modal Concurrency Gate

A Modal ingress concurrency gate was tested to see whether limiting request admission to match vLLM max-seqs would reduce queue pressure.

This did not improve the score. The result suggested that vLLM's internal scheduler handled the workload better than the external gate for this configuration.

## Compiled Mode

Compiled mode improved inter-token latency but worsened time to first token and reduced throughput enough that the final score stayed below the eager b2048 configuration.

This was an important transfer-learning result: a technique that helps text-heavy settings does not automatically improve mixed multimodal serving.

## Disaggregated Prefill/Decode Prototype

A disaggregated serving prototype was attempted as the most ambitious architecture change:

![Attempted disaggregated prefill/decode architecture](report_assets/architecture_disaggregated.svg)

The goal was to separate prefill and decode roles and transfer KV state through LMCache/NIXL. The prototype did not become score-producing because the available LMCache integration was incompatible with the installed vLLM version.

The companion document DISAGGREGATED_SERVING_DETAILS.pdf explains this attempt in more detail.

## Overall Design Lessons

| Method | Outcome | Lesson |
|---|---|---|
| Scheduler tuning | Accepted | Best gains came from balancing prefill/decode pressure |
| Prefix cache | Accepted | Helpful for this workload |
| Chunked prefill | Accepted | Important for mixed prompt scheduling |
| Request routing 2+2 | Rejected | Heavy pool became constrained |
| Request routing 1+3 | Rejected for final score | Better than 2+2 but still below monolithic |
| N-gram speculation | Rejected | Overhead outweighed benefit in this workload |
| Online FP8 quantization | Rejected | Online quantization did not improve score |
| Modal concurrency gate | Rejected | External admission limiting hurt performance |
| Compiled mode | Rejected for final score | Improved ITL but worsened TTFT and total score |
| Disaggregated serving | Not completed | Dependency alignment blocked fair evaluation |

## Conclusion

The strongest architecture for this study was not the most complex one. The best result came from a well-tuned monolithic vLLM deployment with carefully balanced scheduler settings. The novel methods were still useful because they established what did not transfer cleanly to the mixed multimodal workload and identified future directions for deeper serving-system work.