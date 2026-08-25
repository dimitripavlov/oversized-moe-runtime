# Oversized MoE Runtime

A small policy and launcher layer for running sparse Mixture-of-Experts GGUF models that are larger than physical RAM.

The runtime uses `llama.cpp` as the inference engine. It does not implement its own inference kernels or HTTP server.

## Goal

The main use case is:

```text
large total model capacity
×
small active MoE footprint
×
physical RAM oversubscription
```

The runtime automatically inspects the GGUF model and host memory, chooses a bounded expert-residency budget, and launches `llama-completion` or `llama-server` with the appropriate oversized-model memory policy.

## Current status

MVP 0.1 currently supports the architectures validated by this project:

* `qwen3moe`
* `qwen3next`

The current low-level ExpertResidency implementation supports models with:

* 128 experts per layer
* 512 experts per layer

Validated models include:

* Qwen3-30B-A3B Q4_K_M
* Qwen3-Next-80B-A3B-Instruct Q4_K_M

Qwen3-Next-80B-A3B-Instruct Q4_K_M is approximately 48.4 GB and has been successfully run CPU-only on a 16 GB Apple M1 system without swap.

## Build

Configure `llama.cpp` with server support:

```bash
cmake -S . -B build \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_SUBPROCESS=ON
```

Build the runtime and engines:

```bash
cmake --build build \
  --target \
    oversized-moe \
    oversized-moe-probe \
    oversized-moe-run \
    oversized-moe-serve \
    llama-completion \
    llama-server \
  -j4
```

The main user-facing binary is:

```text
build/bin/oversized-moe
```

The lower-level binaries remain available as internal/runtime components:

```text
build/bin/oversized-moe-probe
build/bin/oversized-moe-run
build/bin/oversized-moe-serve
```

## Commands

### Inspect a model

```bash
./build/bin/oversized-moe probe MODEL.gguf
```

The probe reports:

* GGUF architecture
* model size
* layer count
* expert count
* active experts
* expert tensor geometry
* physical RAM
* available RAM estimate
* oversubscription ratio
* expert residency budget
* estimated experts-per-tensor quota
* mmap policy
* full-prefetch policy
* repack policy

Example:

```bash
./build/bin/oversized-moe probe \
  ~/Models/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf
```

### Run completion

```bash
./build/bin/oversized-moe run \
  MODEL.gguf \
  -t 4 \
  -c 512 \
  -n 64 \
  --temp 0 \
  -p "Explain why the sky is blue."
```

The remaining arguments are passed through to `llama-completion`.

To inspect the generated command without starting inference:

```bash
./build/bin/oversized-moe run \
  --dry-run \
  MODEL.gguf \
  -t 4 \
  -c 512 \
  -n 64 \
  -p "Hello"
```

### Run server

```bash
./build/bin/oversized-moe serve \
  MODEL.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  -t 4 \
  -c 2048
```

The remaining arguments are passed through to `llama-server`.

Dry-run:

```bash
./build/bin/oversized-moe serve \
  --dry-run \
  MODEL.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  -t 4 \
  -c 2048
```

After the server is ready:

```bash
curl -sS http://127.0.0.1:8080/health
```

Expected response:

```json
{"status":"ok"}
```

`llama-server` provides its normal OpenAI-compatible API, including `/v1/chat/completions`.

Example:

```bash
curl -sS \
  http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Explain in a few sentences why the sky is blue."
      }
    ],
    "temperature": 0,
    "max_tokens": 32
  }'
```

## Automatic memory policy

For an oversized supported sparse-MoE model, the runtime automatically controls:

```text
mmap
full mmap prefetch policy
repack policy
expert residency budget
experts-per-tensor residency quota
```

Internally, the current engine integration uses:

```text
GGML_EXPERT_RESIDENT_PER_TENSOR
```

but users should not set this variable manually.

The runtime computes the quota from:

```text
physical RAM
model size
oversubscription ratio
expert tensor geometry
lockable bytes per expert residency slot
global RAM safety reserve
```

The policy is designed to preserve RAM for the complete inference working set rather than maximize expert-cache hit rate by itself.

This is important because a larger expert cache and a higher expert-cache hit rate do not necessarily produce higher total inference throughput when physical RAM is constrained.

## Runtime-owned options

The following options are controlled by the oversized runtime and should not be supplied manually:

```text
-m
--model

-lm
--load-mode

--mmap
--no-mmap

--repack
--no-repack

--mlock
```

Conflicting values are rejected before the engine is started.

Normal inference and server options remain passthrough arguments, including:

```text
threads
batch threads
context size
generation length
temperature
prompt
host
port
```

## Runtime validation

Before starting `llama-completion` or `llama-server`, the runtime validates the generated policy.

Checks include:

* model file readability
* engine executable availability
* valid physical RAM information
* supported sparse-MoE architecture
* valid expert residency geometry
* expert residency quota not exceeding expert count
* expert residency budget fitting the global physical-RAM budget
* mmap enabled for oversized mode
* full mmap prefetch disabled for oversized mode
* repack disabled for oversized mode

A valid configuration reports:

```text
Runtime validation
------------------
Runtime policy:     OK
```

The engine is not started if the generated runtime policy is internally inconsistent.

## Unsupported models

MVP 0.1 intentionally uses conservative fallback behavior.

A dense model is rejected as not being a supported sparse-MoE model.

An unknown sparse-MoE architecture is rejected instead of guessing its expert tensor layout or residency policy.

A supported sparse-MoE model that fits physical RAM uses standard mode and normal engine defaults rather than forcing oversized-model residency.

## Validated operating points

### Qwen3-30B-A3B Q4_K_M

Validated on a 16 GB Apple M1 system:

```text
model size:          ~18.6 GB
architecture:        qwen3moe
layers:              48
experts/layer:       128
active experts:      8
expert tensors:      144
automatic quota:     48 experts/tensor
expert budget:       ~6.08 GiB
```

The runtime probe measures approximately:

```text
raw bytes/quota:      130.78 MiB
lockable bytes/quota: 128.53 MiB
```

### Qwen3-Next-80B-A3B-Instruct Q4_K_M

Validated on a 16 GB Apple M1 system:

```text
model size:          ~48.4 GB
architecture:        qwen3next
layers:              48
experts/layer:       512
active experts:      10
expert tensors:      144
oversubscription:    ~2.82x
automatic quota:     32 experts/tensor
expert budget:       ~2.66 GiB
```

The runtime probe measures approximately:

```text
raw bytes/quota:      87.19 MiB
lockable bytes/quota: 84.94 MiB
```

A real Qwen3-Next run with automatic quota 32 produced approximately:

```text
locked expert pages: 2718 MiB
lock failures:       0
```

The Qwen3-Next model has been validated through both:

```text
oversized-moe run
oversized-moe serve
```

including:

* successful CPU-only inference
* successful `llama-server` startup
* successful `/health` check
* successful OpenAI-compatible `/v1/chat/completions` request

## Design principles

The runtime follows several deliberate constraints.

### Keep llama.cpp as the inference engine

The project does not attempt to replace `llama.cpp`.

The product layer is responsible for:

```text
model inspection
memory policy
MoE residency policy
launch configuration
runtime validation
```

`llama.cpp` remains responsible for:

```text
model execution
CPU kernels
token generation
KV/state management
sampling
HTTP serving
```

### Optimize global physical RAM, not cache hit rate

Expert-cache hit rate is useful diagnostic information, but it is not the product objective.

The relevant objective is:

```text
maximum useful inference throughput
under the complete physical-RAM budget
```

RAM must remain available for:

```text
expert residency
shared experts
dense/nonexpert weights
attention or DeltaNet state
OS file cache
runtime buffers
KV/state
```

### Prefer safe refusal over unsupported guessing

The runtime does not attempt to infer arbitrary MoE layouts from incomplete metadata.

New architectures should be explicitly validated before automatic residency policy is enabled for them.

## Scope of MVP 0.1

MVP 0.1 is intentionally small.

It does not currently attempt to provide:

* a new inference engine
* custom matrix kernels
* a custom HTTP server
* adaptive expert eviction
* asynchronous expert-prefetch workers
* automatic optimization for arbitrary MoE architectures
* maximum expert-cache hit rate at the expense of the rest of the RAM working set
* automatic model downloading
* package installation outside the current `llama.cpp` build

The product layer is responsible for safe oversized sparse-MoE memory execution policy.

`llama.cpp` remains responsible for inference.

