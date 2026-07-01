# SYCL Dual-GPU Optimization Runbook

This runbook is for continuing SYCL backend performance work once a dual-GPU machine is available.

## Goal

Improve end-to-end throughput and latency for dual-GPU inference by reducing synchronization stalls, avoiding unnecessary data movement, and improving split/load balance.

## Scope

Focus on SYCL backend behavior in these areas:

1. Event-based de-serialization (remove blocking waits where safe).
2. Size-aware and topology-aware device-to-device transfer policy.
3. Better split scheduling for asymmetric GPUs.
4. Shared-memory buffer reuse where legal.
5. Lightweight timing instrumentation to identify bottlenecks.

## Current State Snapshot

Recent work already merged on branch sycl-host-usm-shared-mode:

- Host/shared-memory mode enablement in SYCL backend.
- Automatic integrated/shared-memory detection when GGML_SYCL_USE_HOST_USM is unset.
- Shared-memory mode now uses sycl::malloc_shared for main shared path.

Use this as baseline before starting dual-GPU tuning.

## Required Hardware

Minimum:

- 2 SYCL-visible GPUs in one system.
- Prefer one Intel dGPU + one Intel iGPU for asymmetric test coverage.

Nice-to-have:

- 2 similar dGPUs for symmetric scaling comparison.
- PCIe topology visibility (to interpret transfer behavior).

## Required Software

- OneAPI compiler/runtime installed and working.
- Level Zero runtime working for target GPUs.
- llama.cpp built with SYCL backend enabled.

Recommended build flags:

- GGML_SYCL=ON
- GGML_SYCL_F16=ON (and compare with OFF)
- GGML_SYCL_GRAPH=ON (and compare with disabled runtime flag)

## Runtime Knobs to Record in Every Run

Always log these values:

- GGML_SYCL_USE_HOST_USM
- GGML_SYCL_USE_LEVEL_ZERO_API
- GGML_SYCL_DEV2DEV_MEMCPY
- GGML_SYCL_DISABLE_GRAPH
- GGML_SYCL_USE_ASYNC_MEM_OP
- GGML_SYCL_DISABLE_DNN
- GGML_SYCL_ENABLE_VMM
- GGML_SYCL_USM_SYSTEM

Also capture:

- Number of SYCL devices detected
- Main device
- Split mode and tensor/layer split values

## Baseline Benchmark Protocol

Run each test at least 3 times and report median.

Metrics:

- Prompt processing tokens/s
- Decode tokens/s
- Time to first token
- End-to-end latency for fixed token count
- GPU utilization per device
- Copy engine utilization per device
- Host RAM and GPU memory usage

Test matrix:

1. Single-GPU baseline (GPU0 only).
2. Single-GPU baseline (GPU1 only).
3. Dual-GPU current implementation (no code changes).
4. Dual-GPU with host/shared mode forced off (GGML_SYCL_USE_HOST_USM=0).
5. Dual-GPU with host/shared mode forced on (GGML_SYCL_USE_HOST_USM=1).
6. Dual-GPU auto mode (GGML_SYCL_USE_HOST_USM unset).

Use at least two model sizes:

- Small/medium model (to expose overhead sensitivity).
- Larger model (to expose bandwidth and memory pressure behavior).

## Work Plan

### Phase 1 - Event-based de-serialization

Objective:

- Replace blocking queue waits in hot transfer/copy paths with event dependencies.

Targets:

- Paths currently calling queues_wait_and_throw before host-visible access.
- Cross-device copy paths that serialize producer and consumer unnecessarily.

Validation:

- Functional parity (no output corruption).
- Reduced queue idle time.
- Improved overlap between copy and compute.

### Phase 2 - Transfer policy tuning

Objective:

- Choose transfer method by payload size and topology characteristics.

Targets:

- Existing dev2dev_memcpy decision points.
- Distinguish small/medium/large transfer behavior.

Validation:

- Lower transfer time percentiles.
- Better dual-GPU scaling on large decode batches.

### Phase 3 - Split/load balancing

Objective:

- Improve tensor split and row split decisions for asymmetric GPUs.

Targets:

- Split heuristics based on observed per-device throughput.
- Avoid overloading slower GPU in mixed iGPU+dGPU setup.

Validation:

- Lower tail latency per token.
- Better aggregate throughput than static split.

### Phase 4 - Shared-memory buffer reuse

Objective:

- Reuse temporary/work buffers safely when shared-address memory is available.

Targets:

- Temporary allocations in reorder and intermediate copy paths.
- Lifetime-safe reuse only (no aliasing races).

Validation:

- Reduced allocation churn.
- Stable memory footprint and no regressions in correctness.

### Phase 5 - Instrumentation

Objective:

- Add minimal timing counters for critical stages.

Counters to add:

- Per-device copy time
- Per-device kernel time
- Per-device wait/barrier time
- Alloc/free overhead in hot loops

Validation:

- Actionable breakdown explaining where time is spent.

## Correctness Guardrails

For every optimization patch:

1. Compare generated output text against baseline for deterministic settings.
2. Run with sanitizing/debug logs where available.
3. Stress test with long generation and mixed prompt sizes.
4. Test both shared-memory mode on and off.

If corruption appears:

- Reintroduce strict ordering at the last touched boundary.
- Narrow to first incorrect tensor transfer using added timing/log points.

## Suggested Execution Order

1. Instrument first (low risk, high observability).
2. Event de-serialization.
3. Transfer policy tuning.
4. Split/load balancing.
5. Shared-memory buffer reuse.

## Reporting Template

For each experiment, record:

- Commit SHA
- Hardware (GPU model pair)
- Driver/runtime versions
- Build flags
- Runtime env vars
- Model and prompt settings
- Median metrics and variance
- Regression check result (pass/fail)

## Exit Criteria

Accept optimization set when all are true:

- No correctness regressions across test matrix.
- Dual-GPU throughput gain vs baseline is measurable and repeatable.
- Time-to-first-token does not regress materially for small prompts.
- Memory footprint remains stable or improves.

## Notes

- Shared USM on dGPU may trigger migration behavior; this can be better or worse depending on access pattern.
- Do not assume copy-engine reduction alone means end-to-end speedup; compute overlap and queue stalls are usually decisive.
