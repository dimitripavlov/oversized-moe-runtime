# Oversized MoE Runtime — Rust product-layer PoC

This directory contains the Rust product-layer proof of concept introduced in
Oversized MoE Runtime v0.2.

The Rust layer does not implement inference kernels, tensor execution, or the
hot ExpertResidency mechanism. Those remain in llama.cpp / ggml C/C++.

## Status

v0.2 implements:

    oversized-moe-rs probe MODEL.gguf
    oversized-moe-rs run MODEL.gguf [llama-completion args...]

Rust `serve` is not implemented in v0.2. The C++ frontend remains the complete
reference implementation, including `serve`.

## Requirements

The Rust frontend requires:

- a Rust toolchain with Cargo and Rust 2024 edition support;
- no external Cargo dependencies for the v0.2 crate.

Actual inference through `run` additionally requires a compatible
`llama-completion` binary built from this patched llama.cpp tree.

Primary validation is currently macOS arm64 / Apple M1, CPU-only.

## Build

From the repository root:

    cargo build --release \
      --manifest-path tools/oversized-moe-rs/Cargo.toml

The resulting binary is:

    tools/oversized-moe-rs/target/release/oversized-moe-rs

For actual inference, build llama-completion as well:

    cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_METAL=OFF

    cmake --build build \
      --target llama-completion \
      -j4

## Probe a model

    ./tools/oversized-moe-rs/target/release/oversized-moe-rs \
      probe MODEL.gguf

The Rust probe reads GGUF metadata directly and applies the same product policy
as the C++ implementation.

On the validated 16 GiB policy baseline:

    Qwen3-30B-A3B       -> quota 48
    Qwen3-Next-80B-A3B -> quota 32

## Dry-run

Inspect the launch plan without starting inference:

    ./tools/oversized-moe-rs/target/release/oversized-moe-rs \
      run --dry-run MODEL.gguf \
      -t 4 \
      -c 512 \
      -n 64

## Run inference

    ./tools/oversized-moe-rs/target/release/oversized-moe-rs \
      run MODEL.gguf \
      -t 4 \
      -tb 4 \
      -c 512 \
      -n 64 \
      --temp 0

On Unix the validated product path uses exec() to replace the Rust launcher
with llama-completion after probing, policy selection and validation.

## Engine discovery

If llama-completion is not discovered automatically, specify it explicitly:

    LLAMA_COMPLETION_BIN="$PWD/build/bin/llama-completion" \
      ./tools/oversized-moe-rs/target/release/oversized-moe-rs \
      run MODEL.gguf -n 64

## Engine contract

For oversized mode the Rust product layer uses the generic llama.cpp controls:

    -lm mmap
    --no-mmap-prefetch
    --no-repack
    --expert-residency-per-tensor N

The product layer owns `N`.

Users cannot override runtime-owned model-loading, mmap-prefetch, repack,
mlock, or expert-residency controls through passthrough arguments.

For deterministic behavior the launcher removes inherited values of:

    GGML_EXPERT_RESIDENT_PER_TENSOR
    LLAMA_ARG_EXPERT_RESIDENCY_PER_TENSOR

## Validation

    cargo test --release \
      --manifest-path tools/oversized-moe-rs/Cargo.toml

    cargo check \
      --manifest-path tools/oversized-moe-rs/Cargo.toml

The v0.2 crate currently has no dedicated Rust unit/integration test suite, so
cargo test executes zero Rust tests. Adding meaningful Rust tests is part of
the planned v0.3 productization stage.

The C++ Oversized MoE regression suite remains the v0.2 reference and contains
exactly seven test targets.

## Architecture boundary

The Rust frontend deliberately does not depend on internal llama.cpp / ggml
C++ types such as:

    ggml_tensor
    op_params
    internal graph objects

Hot-path functionality remains in C/C++:

    llama.cpp inference
    ggml tensor execution
    mmap-backed model access
    bounded zero-copy ExpertResidency
    mlock / munlock residency management

## Current limitations

- Rust `serve` is not implemented in v0.2.
- Rust does not yet have a dedicated regression-test suite.
- Primary validation is macOS arm64 / Apple M1.
- Low-level ExpertResidency is currently Apple-specific.
- Product support is currently limited to `qwen3moe` and `qwen3next`.
- A patched llama.cpp engine is still required.

See the root README.md and BENCHMARKS.md for project-level documentation and
performance results.
