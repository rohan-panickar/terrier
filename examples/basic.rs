//! Retry a flaky operation with the default decorrelated-jitter backoff.
//!
//! Run with: `cargo run --example basic`
//!
//! `charge_card` fails its first two attempts with a transient error, then
//! succeeds. `.when` retries only the transient variant; `.notify` prints each
//! backoff so you can watch the schedule widen.

use std::time::Duration;
use terrier::{Retry, RetryInfo};

#[derive(Debug)]
#[allow(dead_code)] // `Declined` documents the permanent path; the happy demo never hits it.
enum ChargeError {
    /// The payment gateway is briefly unreachable — worth retrying.
    GatewayUnavailable,
    /// The card was declined — permanent, never retry.
    Declined,
}

impl ChargeError {
    fn is_transient(&self) -> bool {
        matches!(self, Self::GatewayUnavailable)
    }
}

impl std::fmt::Display for ChargeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GatewayUnavailable => f.write_str("payment gateway unavailable"),
            Self::Declined => f.write_str("card declined"),
        }
    }
}

#[tokio::main]
async fn main() {
    let mut n = 0u32;
    let result = Retry::new()
        .max_attempts(5)
        .initial_backoff(Duration::from_millis(20))
        .max_backoff(Duration::from_millis(200))
        .attempt(move || {
            let attempt = n;
            n += 1;
            async move {
                if attempt < 2 {
                    Err(ChargeError::GatewayUnavailable)
                } else {
                    Ok::<u64, ChargeError>(4_200)
                }
            }
        })
        .when(ChargeError::is_transient)
        .notify(|err: &ChargeError, info: RetryInfo| {
            println!(
                "attempt {} failed ({err}); retrying in {:?}",
                info.attempt_index, info.delay
            );
        })
        .await;

    match result {
        Ok(cents) => println!("charged {cents} cents"),
        Err(ChargeError::Declined) => println!("gave up: card declined (permanent)"),
        Err(err) => println!("gave up: {err}"),
    }
}
