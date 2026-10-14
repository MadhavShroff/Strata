# Strata on NVIDIA DGX Spark (aarch64, GB10): measured results

Branch `sparky/aarch64-port`. Measured on 2026-10-14 on one DGX Spark
("Sparky"). These are first measurements, not a tuned benchmark.

## Machine and model

| Item | Value |
| --- | --- |
| Machine | NVIDIA DGX Spark, GB10 Grace Blackwell, compute capability 12.1 |
| Memory | 128 GB unified (CPU and GPU share one memory), 121.7 GB visible |
| CPU | 20 Arm cores (10 Cortex-X925, 10 Cortex-A725); no x86 features, so CPU experts use ggml-cpu (NEON) |
| CUDA | 13.0 |
| Model | Qwen3.8-Flash-Next, GSQ-RCO IQ2_XS pack, two GGUF shards, 64 GiB |
| Drafter | MTP draft layer (`--mtp`), speculative decoding on |
| Reasoning | ON in every run, reasoning effort high (never reduced for speed) |
| Sampling | greedy (temperature 0) |

The job ran under a memory-capped systemd unit (GATE admission on Sparky).

## Context limit

| Setting | Value |
| --- | --- |
| `--max-context` | 262144 tokens (the model's trained context) |
| KV cache | int8, 32768 cells per layer resident on the GPU, the rest in pinned RAM |
| Rope scaling | none (native): no `--rope-scaling` or `--yarn-*` flag was given |

## Decode speed, one request at a time

Prompt: a 79-token request for a technical explanation of B-tree balancing.
Each run generates 2000 tokens. The speed is the engine's own figure from
its `N generated in M ms (X tok/s)` line.

### Clean baseline: every expert resident on the GPU

Profile `resident`: `--expert-cache 24576` (all 24576 expert slots,
33.02 GiB), `--spec 4 --spec-min-p 0.5`, memory cap 70 GiB, 8 CPUs.
One client only.

| Run | Decode tok/s | Prompt tok/s | Drafts accepted |
| --- | --- | --- | --- |
| r1 (cold) | 41.5 | 20.1 | 989 of 1323 (75%) |
| r2 | 52.2 | 112.2 | 991 of 1313 (75%) |
| r3 | 50.4 | 112.9 | 992 of 1319 (75%) |

Median decode: **50.4 tok/s**. Range 41.5 to 52.2.

### Variants (not clean)

Two test scripts ran at the same time against the same server, from
21:46 to 21:57. Each script restarted the server with its own flags, so
some runs in this table used another script's flags or shared the GPU.
Treat these as indications only; they need a clean repeat.

| Variant | Flags | Runs | Decode tok/s, median (range) | Draft acceptance |
| --- | --- | --- | --- | --- |
| lookup3 | `--lookup-chain 3` | 3 | 45.2 (42.1-50.1) | 74% |
| shstream | `STRATA_SH_STREAM=1` | 3 | 50.6 (41.9-52.5) | 76% |
| spec6 | `--spec 6` | 5 | 52.0 (39.9-54.1) | 69% |
| mtpq4 | `--mtp-q4 all` | 0 | did not reach /health in 900 s | - |

No variant clearly beats the baseline.

## Partial expert cache (memory cap 50 GiB)

The first run used a 50 GiB cap, 3 CPUs and `--expert-cache 8446` (34.4%
of the experts; 2.9 GiB of the cache was usable). Experts that missed the
cache ran on the CPU through ggml-cpu (NEON).

- One request, 66 output tokens, reasoning on: 20.4 s wall time. The answer
  was correct.
- Four parallel requests (`--batch 4`): all four answers were correct. They
  finished in 76-86 s, about 1.5 tok/s per request and 5 tok/s in total.
- One decode step took 584 ms, and the CPU expert work took most of it.

Conclusion: on this machine, keep every expert resident on the GPU. The
unified memory gives no benefit to a partial cache, because the CPU
expert path is slow.

## YaRN (extended context): not measured

A quality test was written but not run. Its plan:

- Arms: native, native again (noise floor), and YaRN
  `--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144` (about 1M).
- Method: teacher-forced comparison on the same text, KL divergence and
  top-1 agreement per position, over six short prompts (code, math, fact,
  reasoning, Chinese, chat), 48 new tokens each.

No YaRN number exists yet. The 1M window is not taken until this test
shows no quality loss.

## Correctness

Every answer that was checked was correct (37 x 43 = 1591, and 12-15 x 17
in the parallel test). No quality benchmark was run.

## Reproduce

```
strata --serve --pack <packs/iq2_xs> --native <IQ2_XS shard 1> \
  --ple-gguf <IQ2_XS shard 2> --expert-profile data/expert-profile.bin \
  --expert-cache 24576 --prefill auto --spec 4 --spec-min-p 0.5 \
  --mtp <mtp/rt> --max-context 262144 --kv int8 --kv-resident 32768 \
  --mmap-experts
```
