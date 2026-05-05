# Warmup Deadline Enforcement

**Date:** 2026-05-05
**Status:** Approved

## Problem

When using the ShareGPT dataset in conversation (multi-turn) mode, each warmup "request" is a full conversation — multiple sequential HTTP calls. The current duration-based warmup (`warmup_duration`) only checks the deadline *before* starting a new conversation, not while one is in-flight. After `sleep(warmup_dur).await`, the code awaits all handles, which blocks until every in-flight conversation finishes. This can exceed the configured warmup time bound significantly.

## Solution

After the warmup deadline fires, abort all in-flight warmup task handles before awaiting them. `JoinHandle::abort()` cancels the task at its next `await` point (typically mid-stream on an HTTP SSE chunk read). Since warmup results are always discarded, abrupt cancellation is safe and correct.

## Change Sites

All three warmup sites in `src/benchmark.rs` use `warmup_duration` and have the same fix pattern:

### `run_concurrent_mode_internal`

```rust
// Before
sleep(warmup_dur).await;
for handle in warmup_handles {
    let _ = handle.await?;
}

// After
sleep(warmup_dur).await;
for handle in warmup_handles {
    handle.abort();
    let _ = handle.await;
}
```

### `run_qps_mode_internal`

```rust
// Before
for handle in handles.drain(..) {
    let _ = handle.await?;
}

// After
for handle in handles.drain(..) {
    handle.abort();
    let _ = handle.await;
}
```

### `run_saturation_mode_internal`

```rust
// Before
sleep(warmup_dur).await;
for handle in warmup_handles {
    let _ = handle.await;
}

// After
sleep(warmup_dur).await;
for handle in warmup_handles {
    handle.abort();
    let _ = handle.await;
}
```

## Properties

- **No config changes** — enforces the existing `warmup_duration` field.
- **No new dependencies** — `JoinHandle::abort()` is in tokio.
- **Error handling** — `handle.await` after abort returns `Err(JoinError::Cancelled)`, which all three sites already ignore via `let _ = handle.await`.
- **Warmup count accuracy** — the `warmup_completed` counter will only reflect conversations that finished before the deadline, which is correct.
