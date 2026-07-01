# SYCL Integrated Arc Recommendations

This document tracks practical optimization steps for Intel integrated Arc + SYCL.

## Goals

1. Reduce memory duplication in shared-memory mode.
2. Reduce kernel stalls in decode hotspots.
3. Keep changes small and measurable with VTune.

## Current Baseline

1. Shared-memory mode is enabled by `GGML_SYCL_USE_HOST_USM` and uses `sycl::malloc_shared`.
2. Reorder MMVQ kernels were tuned (smaller workgroups + lower inner-loop overhead).
3. FP16 flash-attn `ncols=2` path reduced `nbatch_fa` to reduce register spills.
4. Split tensor extras now use shared USM in host-USM mode.

## Recommended Next Steps

### Step 1: Continue flash-attn spill reduction

1. Tune FP16 flash-attn config for `ncols=4` and `ncols=8` to reduce spill/stall pressure.
2. Keep launch shape and algorithm unchanged; only tile config values should change.
3. Validate with VTune metrics: Spill Memory Size, Active, Stalled, total kernel time.

### Step 2: Reduce synchronization serialization on host-visible paths

1. Audit blocking `queues_wait_and_throw()` calls in sync set/get/copy paths.
2. Replace unnecessary full-queue waits with event-ordered copies where safe.
3. Keep strict ordering in correctness-sensitive boundaries.

### Step 3: Add size-aware copy policy in shared-memory mode

1. Introduce copy size thresholds for host-visible buffers.
2. Use direct CPU memcpy for very small copies.
3. Use queue memcpy for medium and large copies.

### Step 4: Expand memory dedup in shared mode

1. Prefer shared allocations for immutable tensor residency where feasible.
2. Avoid temporary duplicate backing for read-only tensors in single-GPU flows.
3. Keep fallback path for unsupported tensor layouts.

## Measurement Protocol

For each step, capture before/after:

1. Prompt and decode tokens/s.
2. Top 5 GPU kernels by total time.
3. For each top kernel: Active, Stalled, Idle, Spill Memory Size.
4. Process memory usage and GPU memory usage.

## Commit Policy

1. Use one commit per improvement step.
2. Keep each commit focused and reversible.
3. Attach VTune delta notes in commit message body or follow-up notes.
