# Changelog

## 0.2.1 - 2026-06-11

### Fixed

- **Honored hints are now bounded by `max_backoff` by default.** Previously, a
  `.delay_hint` return value was clamped only by the one-year `SAFE_SLEEP_CAP`,
  so a garbage `Retry-After` (e.g. an epoch timestamp parsed as seconds) would
  park a retry for up to a year. With `Retry`, the honored hint is now capped at
  `max_backoff` unless you explicitly raise it. Custom-policy callers (`Fixed`,
  closures, user `Policy` impls) are unaffected: their `max_delay_hint` seeds as
  `None`, preserving the previous one-year-cap behavior.

### New

- **`.max_delay_hint(Duration)`** on `RetryFuture`. Sets the ceiling on an honored
  hint independently of policy backoff. `Retry::attempt` seeds it to `max_backoff`
  by default; raise it (e.g. `.max_delay_hint(Duration::from_secs(300))`) to honor
  a longer server-supplied delay without changing the exponential backoff ceiling.

---

## 0.2.0 - 2026-06-11

Breaking reshape of the public API. No existing consumers; safe to break.

### Breaking changes

- **`Backoff` enum replaces `jitter`/`multiplier` fields.** The dead
  `jitter: bool` + `multiplier: f64` combination is gone. Use
  `.decorrelated_jitter()` (the default) or `.exponential(m)` instead.
  `.jitter(false).multiplier(2.0)` → `.exponential(2.0)`.
- **`Policy::next_delay` is the sole required method.** `Policy::delay` is
  removed. `fn next_delay(&self, attempt: u32, previous: Option<Duration>) -> Duration`
  is the contract; stateless policies can ignore `previous`.
- **`Policy::caps()` → `Policy::budget() -> Budget`.** `Budget` is a
  `#[non_exhaustive]` struct with `max_attempts: u32` and
  `max_elapsed: Option<Duration>`. Construct it with `Budget::new(n)` or
  `Budget::new(n).max_elapsed(d)`. Update any `caps()` overrides to
  return a `Budget`.
- **`Retryable` extension trait removed.** Drop `use terrier::Retryable`.
  Use `Retry::new()...attempt(op)` (or `policy.attempt(op)`) instead.
- **`RetryInfo::attempt` → `RetryInfo::attempt_index`.** Now 0-based.
  Update `.notify` callbacks that read `info.attempt`.
- **`Retry::delay_for_attempt` demoted to `pub(crate)`.** It returned
  non-reproducible samples under jitter anyway.

### New

- **`pub async fn RetryFuture::run()`** is now public. Use
  `tokio::spawn(retry.run())` when you need to spawn the loop.
- **`DEFAULT_MAX_ATTEMPTS: u32 = 3`** is now a public constant.
- **O(1) closed-form exponential delay** replaces the old iterative loop;
  `multiplier(1.0 + EPSILON)` + `attempt = u32::MAX` no longer spins.
- **`DecorrelatedJitter::sample_with`** private seam preserved for the AWS
  oracle test.
- **The executor now feeds the actual previous sampled sleep forward** as
  `previous` to `Policy::next_delay`, enabling the true AWS decorrelated
  recurrence `sleep = min(max_backoff, random(base, prev_sleep * 3))`.
  In 0.1.0 the executor reconstructed `prev` statelessly per attempt; this
  is a jitter-distribution change.
- Added new coverage tests: BLOCK spin regression, `base > prev*3` edge,
  zero-base jitter, `jitter_fraction` uniformity and multi-thread distinctness,
  `Budget` struct, `run()` direct call.

### Dependencies

- `tokio` (`time` feature) — unchanged.

---

## 0.1.0 - 2026-06-11

First release. Tenacious async retries for Rust: bounded backoff with
decorrelated jitter, retrying until the operation succeeds or the configured
budget says stop.

### Features

- **Fluent builder.** `Retry::new()` configures the schedule; `.attempt(op)`
  attaches the operation and returns an awaitable `RetryFuture`. `Retry` is `Copy`.
- **Decorrelated-jitter backoff by default.** AWS-style recurrence
  `sleep = min(max_backoff, random(base, base * 3^attempt))` — the previous
  delay was reconstructed statelessly from the attempt index, not fed forward.
- **Error classification.** `.when(predicate)` retries only transient errors.
- **Per-attempt operations.** `.attempt_with(|attempt: u32| ...)`.
- **Pluggable strategies.** `Policy` trait; `Fixed`, `DecorrelatedJitter`,
  and `Fn(u32) -> Duration` built in.
- **Server-hint honoring.** `.delay_hint(|e| Option<Duration>)`.
- **Observation hook.** `.notify(fn)` with `RetryInfo`.
- **Wall-clock budget.** `max_elapsed` with a pre-sleep check.
- **Safe far-future sleeps.** All delays clamped to one year.
