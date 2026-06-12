# terrier

Tenacious async retries for Rust: decorrelated-jitter backoff, an attempt cap, and an optional wall-clock budget.

## Install

```sh
cargo add terrier
```

Depends only on `tokio` (`time` feature). It emits no logs of its own; use `.notify()` to record retries.

## Quickstart

Configure a `Retry`, attach an operation, classify your errors with `.when`, and `.await` it.

```rust
use terrier::Retry;
use std::time::Duration;

#[derive(Debug)]
enum DbError { SerializationConflict, Deadlock, ConstraintViolation(String) }

impl DbError {
    fn is_transient(&self) -> bool {
        matches!(self, Self::SerializationConflict | Self::Deadlock)
    }
}

// `commit_transaction` is your own `async fn() -> Result<u64, DbError>`.
async fn run() -> Result<u64, DbError> {
    let rows = Retry::new()
        .max_attempts(5)
        .initial_backoff(Duration::from_millis(50))
        .max_backoff(Duration::from_secs(2))
        .attempt(commit_transaction)
        .when(DbError::is_transient)
        .await?;
    Ok(rows)
}
```

The operation re-runs on every attempt, so it must be safe to repeat (see [Semantics](#semantics)). With no `.when`, every error retries up to `max_attempts`.

## Configuration

| Knob | Default | Meaning |
|---|---|---|
| `max_attempts` | 3 | Total calls, first try plus retries. Panics on 0. |
| `initial_backoff` | 500 ms | Sleep before the second attempt. |
| `max_backoff` | 30 s | Ceiling on any single sleep. |
| `.decorrelated_jitter()` | (default) | AWS decorrelated jitter: `min(max_backoff, random(initial_backoff, prev_sleep * 3))`. |
| `.exponential(m)` | — | Deterministic: `min(max_backoff, initial_backoff * m^attempt)`. Panics if `m < 1.0`. |
| `max_elapsed` | none | Wall-clock budget across all attempts and sleeps. |
| `.when` | retry all | Retry only when the predicate returns `true` for the error. |
| `.notify` | none | Run a callback before each retry sleep. |
| `.delay_hint` | none | Override the backoff with a delay read from the error. |

`.when` takes a function item or a closure:

```rust
Retry::new()
    .attempt(op)
    .when(MyError::is_transient)        // or: .when(|e: &MyError| e.is_transient())
    .await?;
```

To opt into deterministic exponential backoff:

```rust
Retry::new()
    .initial_backoff(Duration::from_millis(100))
    .max_backoff(Duration::from_secs(10))
    .exponential(2.0)
    .attempt(op)
    .await?;
```

## Features

`.notify(fn)` runs after each failed, retryable attempt, just before the sleep. It never fires on success or the final attempt. `RetryInfo { attempt_index, delay }` is `#[non_exhaustive]`; `attempt_index` is 0-based.

```rust
use terrier::{Retry, RetryInfo};

Retry::new()
    .attempt(call_endpoint)
    .when(ApiError::is_transient)
    .notify(|err: &ApiError, info: RetryInfo| {
        eprintln!("attempt {} failed ({err}); retrying in {:?}", info.attempt_index, info.delay);
    })
    .await?;
```

`.attempt_with(fn)` passes the 0-based attempt index, for calls that change per try: an escalating timeout, a widening page, a rotating endpoint.

```rust
Retry::new()
    .attempt_with(|attempt| {
        let timeout = Duration::from_millis(200) * (attempt + 1);  // 200ms, 400ms, ...
        async move { call_with_timeout(timeout).await }
    })
    .when(ApiError::is_transient)
    .await?;
```

`.delay_hint(fn)` lets the server set the wait. When it returns `Some(d)` for a retryable error, `d` replaces the computed backoff for that attempt, honoring an HTTP `Retry-After`. The budget and one-year clamp still bound it. The hinted sleep is also fed back as `previous` to the next policy draw, widening the next decorrelated-jitter window.

```rust
Retry::new()
    .attempt(fetch)
    .delay_hint(|e: &ThrottleError| e.retry_after)
    .await?;
```

`.run()` is public and can be `tokio::spawn`ed directly when `T`, `E`, `F`, `Fut`, and `P` are `Send + 'static`:

```rust
tokio::spawn(retry.run());
```

## Strategies

Schedules live behind the `Policy` trait. The executor owns the attempt cap, the budget, the clamp, and the hooks, so a policy only decides how long to wait. `Retry` is the canonical policy and never makes you name the trait. Also built in:

- `Fixed(d)`: the same delay every attempt.
- `DecorrelatedJitter::new(base, max)`: the standalone jitter policy `Retry` uses.

Any `Fn(u32) -> Duration` is a `Policy`, and a named type can implement one for a stateful schedule:

```rust
use terrier::Policy;
use std::time::Duration;

// Linear backoff: 100ms, 200ms, 300ms, ...
let policy = |attempt: u32| Duration::from_millis(100) * (attempt + 1);
policy.attempt(call_endpoint).max_attempts(4).await?;
```

The sole required method is `next_delay(&self, attempt: u32, previous: Option<Duration>) -> Duration`.
Override `budget()` returning `Budget::new(n)` or `Budget::new(n).max_elapsed(d)` to seed the executor's defaults from your policy's fields.

A bare `Policy` carries no cap or budget; set them on the future with `.max_attempts` / `.max_elapsed`.

## Semantics

**Idempotency.** The operation re-runs on every retry, so only wrap work that is safe to repeat: a transaction that rolls back on failure, an idempotent `PUT`, a deduplicated publish. A non-idempotent write can apply twice.

**Budget.** `max_attempts` bounds tries, not time; ten attempts under a 30 s cap can still block for minutes. `max_elapsed` adds a wall-clock budget, checked before each sleep, so an overrun is bounded by one operation rather than a full backoff.

**Jitter.** With jitter on, sleeps follow the AWS decorrelated-jitter recurrence `sleep = min(max_backoff, random(initial_backoff, prev_sleep * 3))`, seeded at `initial_backoff` and carrying the previous sampled sleep forward. The randomness is a SplitMix64 finalizer over a process counter and wall-clock time: fast and non-cryptographic. `.exponential(m)` gives the deterministic schedule for tests or when you want predictable backoff.

**Sleep ceiling.** Every delay, including `max_backoff(Duration::MAX)`, is clamped to one year before it reaches the timer, so a near-`Duration::MAX` deadline never overflows the timer wheel.

## Examples

Each runs end-to-end against a simulated flaky operation:

```sh
cargo run --example basic          # retry + .when + .notify
cargo run --example custom_policy  # a Fibonacci Policy
cargo run --example server_hint    # honor a Retry-After via .delay_hint
```

## License

MIT
