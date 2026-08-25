# Oversized MoE Runtime Benchmarks

This document records the main measurements that led to
Oversized MoE Runtime v0.1.

The project goal is not simply to minimize model load time or maximize
synthetic cache hit rate. The goal is to make sparse MoE models whose total
GGUF size substantially exceeds physical RAM practically usable on CPU-only
systems.

## Test platform

Primary research and v0.1 validation platform:

    MacBook Air M1
    16 GiB unified RAM
    macOS arm64
    CPU-only
    Metal disabled

Unless stated otherwise, the measurements below were collected on this system.

## Headline result

The largest validated model is:

    Qwen3-Next-80B-A3B-Instruct Q4_K_M
    GGUF size:             48.41 GB
    Physical RAM:          16 GiB
    RAM oversubscription:  ~2.82x
    Architecture:          qwen3next
    Experts/layer:         512
    Active experts/token:  10
    Runtime quota:         32 experts/tensor

The model successfully ran local CPU inference and llama-server on the
16 GiB M1 system.

An OpenAI-compatible `/v1/chat/completions` request was also validated.

## Qwen3-Next-80B-A3B

### Matched decode experiment

A matched workload was used to compare the normal mmap path with bounded
zero-copy expert residency.

| Configuration | Decode throughput |
| --- | ---: |
| Baseline, no expert residency | 3.39 tok/s |
| Expert residency, quota 32 | 4.80 tok/s |

Relative improvement:

    +41.6%

This was the key result demonstrating that retaining a bounded set of hot
expert pages can materially improve oversized sparse-MoE CPU inference.

### Expert locality

Across measured workloads, a 32-entry-per-tensor LRU produced approximately:

    55.8% to 63.8% hit rate

This showed substantial temporal locality in expert routing even with
512 experts per layer.

Increasing cache capacity further did not automatically improve throughput.

One measured comparison was:

    LRU32: 3.80 tok/s
    LRU64: 3.79 tok/s

The higher-capacity cache improved residency coverage but did not improve
end-to-end decode speed.

This is an important design result: maximizing cache hit rate is not the same
as maximizing inference throughput. Physical RAM allocation and memory-system
behavior matter.

## v0.1 clean runtime validation

A later end-to-end run through the product runtime used:

    automatic quota:       32 experts/tensor
    locked residency:      ~2718 MiB
    residency hit rate:    53.39%
    lock failures:         0

Measured decode throughput in that validation run was:

    3.69 tok/s

This number should not be compared directly with the 4.80 tok/s matched
experiment above because the validation workload and run configuration were
different.

The purpose of this run was to validate the complete product path:

    oversized-moe
        -> policy
        -> runtime validation
        -> llama-completion
        -> bounded expert residency

The llama-server path was also validated separately, with measured generation
around:

    ~3.49 tok/s

for that server validation workload.

## Qwen3-30B-A3B

Model:

    Qwen3-30B-A3B Q4_K_M
    Architecture:          qwen3moe
    Experts/layer:         128
    Active experts/token:  8
    GGUF size:             ~18.56 GB

Expert-residency sweep:

| Experts resident / tensor | Decode throughput | Change vs no residency |
| ---: | ---: | ---: |
| 0 | 5.79 tok/s | baseline |
| 16 | 6.76 tok/s | +16.8% |
| 32 | 8.02 tok/s | +38.5% |
| 48 | 8.81 tok/s | +52.2% |

At quota 48 the runtime kept approximately:

    5.99 GiB

of expert pages locked.

This experiment provided the first strong evidence that bounded expert
residency was worth pursuing as a product mechanism.

## What did not improve the result

Several approaches were tested during the research phase and were not carried
into v0.1.

### Larger LRU without regard to RAM allocation

Higher hit rate alone did not guarantee higher throughput.

The LRU32 versus LRU64 result above is the clearest example.

### Explicit prefaulting

Explicitly touching expert pages before use did not provide a useful
end-to-end improvement.

### MADV_WILLNEED

Using `MADV_WILLNEED` as an explicit prefetch hint did not provide a useful
improvement for the target workloads.

### F_RDADVISE

File-level read-ahead advice was also not a productive direction.

### Asynchronous mlock worker

Moving residency locking into an asynchronous worker increased complexity
without delivering the required throughput benefit.

### Copy-based expert cache

A separate cache that copied expert weights was rejected in favor of zero-copy
residency.

The zero-copy approach preserves mmap-backed model storage and controls which
expert pages remain resident rather than creating another copy of expert
weights.

## Design conclusion

The experiments led to the v0.1 architecture:

    GGUF on SSD
         |
         | mmap
         v
    pageable model weights
         |
         | sparse expert accesses
         v
    bounded zero-copy hot-expert residency
         |
         v
    llama.cpp CPU inference

The central observation is that a sparse MoE model does not need all of its
expert weights to be equally resident at the same time.

A sufficiently small active expert set plus routing locality can allow a model
whose total size is much larger than physical RAM to remain practically
usable.

## Reproducibility

v0.1 is distributed as a source release based on the pinned llama.cpp commit:

    f280b26983ad0fdb705a0d9ebf0503e76f2899b0

The release includes:

- the complete source tree;
- a portable 16-patch product series;
- build instructions;
- SHA256 checksums;
- seven automated regression and smoke tests.

The packaged v0.1 source archive was unpacked into an empty directory,
configured from scratch, rebuilt, and passed:

    100% tests passed out of 7

Exact fixed-prompt benchmark harnesses are not yet part of the v0.1 public
release. The throughput measurements in this document are retained from the
research runs that produced the v0.1 design.

A future benchmark package should make workload definitions and raw benchmark
outputs directly reproducible.
