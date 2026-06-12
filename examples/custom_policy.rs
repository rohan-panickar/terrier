//! Plug in a custom backoff schedule via the `Policy` trait.
//!
//! Run with: `cargo run --example custom_policy`
//!
//! Any type that implements `fn next_delay(&self, attempt: u32, previous: Option<Duration>) -> Duration`
//! is a policy. The executor still supplies everything else — the attempt cap, the wall-clock
//! budget, `.when`, `.notify`, `.delay_hint` — so a `Policy` only decides *how long to wait*.
//! Override `budget()` to seed the executor's defaults from the policy's own fields.

use std::time::Duration;
use terrier::Policy;

/// Fibonacci-spaced delays: `unit · {1, 1, 2, 3, 5, 8, …}`.
struct Fibonacci {
    unit: Duration,
}

impl Policy for Fibonacci {
    fn next_delay(&self, attempt: u32, _previous: Option<Duration>) -> Duration {
        let (mut a, mut b) = (1u32, 1u32);
        for _ in 0..attempt {
            (a, b) = (b, a.saturating_add(b));
        }
        self.unit * a
    }

    /// Seed the cap so `Fibonacci { .. }.attempt(op)` already knows when to stop.
    fn budget(&self) -> terrier::Budget {
        terrier::Budget::new(6)
    }
}

#[derive(Debug)]
struct TemporaryFailure;

impl std::fmt::Display for TemporaryFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("temporary failure")
    }
}

#[tokio::main]
async fn main() {
    let mut n = 0u32;
    let outcome: Result<&str, TemporaryFailure> = Fibonacci {
        unit: Duration::from_millis(10),
    }
    .attempt(move || {
        let attempt = n;
        n += 1;
        async move {
            if attempt < 3 {
                Err(TemporaryFailure)
            } else {
                Ok("ok")
            }
        }
    })
    .await;

    println!("{outcome:?}");
}
