//! Honor a server-supplied delay (`Retry-After`) with `.delay_hint`.
//!
//! Run with: `cargo run --example server_hint`
//!
//! When the error carries the server's own backpressure, feed it back into the
//! schedule: a hint wins over the policy's computed delay. The wall-clock budget
//! and the one-year sleep clamp still apply, so a hostile or absurd hint can't
//! blow past `max_elapsed`.

use std::time::Duration;
use terrier::Retry;

#[derive(Debug)]
struct Throttled {
    /// What the server told us to wait, parsed from a `Retry-After` header.
    retry_after: Option<Duration>,
}

impl std::fmt::Display for Throttled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "throttled (retry_after = {:?})", self.retry_after)
    }
}

#[tokio::main]
async fn main() {
    let mut n = 0u32;
    let body = Retry::new()
        .max_attempts(4)
        .initial_backoff(Duration::from_millis(10))
        .attempt(move || {
            let attempt = n;
            n += 1;
            async move {
                if attempt == 0 {
                    Err(Throttled {
                        retry_after: Some(Duration::from_millis(50)),
                    })
                } else {
                    Ok::<String, Throttled>("payload".into())
                }
            }
        })
        .delay_hint(|e: &Throttled| e.retry_after)
        .await;

    println!("{body:?}");
}
