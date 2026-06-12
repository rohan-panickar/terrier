// Paste-check: each test mirrors a README code block; assertion plumbing may replace
// printing. Doc-comment markers and `# ` hidden-line prefixes stripped. If a README
// snippet changes, update here.
#![allow(missing_docs, dead_code)]

use std::time::Duration;
use terrier::{Policy, Retry, RetryInfo};

// ── Quickstart: committing a transaction under serialization conflicts ──────────

#[derive(Debug)]
enum DbError {
    SerializationConflict,
    Deadlock,
    ConstraintViolation(String),
}

impl DbError {
    fn is_transient(&self) -> bool {
        matches!(self, Self::SerializationConflict | Self::Deadlock)
    }
}

async fn commit_transaction() -> Result<u64, DbError> {
    Ok(1)
}

#[tokio::test]
async fn quickstart_commit_transaction() {
    let rows = Retry::new()
        .max_attempts(5)
        .initial_backoff(Duration::from_millis(50))
        .max_backoff(Duration::from_secs(2))
        .attempt(commit_transaction)
        .when(DbError::is_transient)
        .await
        .unwrap();
    assert_eq!(rows, 1);
}

// ── Shared API error for the feature snippets ───────────────────────────────────

#[derive(Debug)]
enum ApiError {
    ServerError(u16),
    Throttled,
    BadRequest(u16),
}

impl ApiError {
    fn is_transient(&self) -> bool {
        matches!(self, Self::Throttled | Self::ServerError(500..=599))
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

async fn call_endpoint() -> Result<String, ApiError> {
    Ok("body".into())
}

// ── Configuration: classifying errors with `.when` (fn-name form) ────────────────

#[derive(Debug)]
enum MyError {
    Transient,
    Permanent,
}

impl MyError {
    fn is_transient(&self) -> bool {
        matches!(self, Self::Transient)
    }
}

async fn op() -> Result<(), MyError> {
    Ok(())
}

#[tokio::test]
async fn case_when_fn_name_form() {
    Retry::new()
        .attempt(op)
        .when(MyError::is_transient)
        .await
        .unwrap();
}

// ── Features: .notify ────────────────────────────────────────────────────────────

#[tokio::test]
async fn case_notify_on_retry() {
    use std::sync::{Arc, Mutex};
    let log: Arc<Mutex<Vec<(u32, Duration)>>> = Arc::new(Mutex::new(Vec::new()));
    let l = Arc::clone(&log);
    let _body = Retry::new()
        .max_attempts(3)
        .initial_backoff(Duration::from_millis(200))
        .attempt(call_endpoint)
        .when(ApiError::is_transient)
        .notify(move |_err: &ApiError, info: RetryInfo| {
            l.lock().unwrap().push((info.attempt_index, info.delay));
        })
        .await
        .unwrap();
}

// ── Features: .attempt_with (per-attempt index) ──────────────────────────────────

struct Response;

async fn call_with_timeout(_timeout: Duration) -> Result<Response, ApiError> {
    Ok(Response)
}

#[tokio::test]
async fn case_attempt_with_per_attempt_index() {
    let _resp = Retry::new()
        .max_attempts(4)
        .initial_backoff(Duration::from_millis(100))
        .attempt_with(|attempt| {
            // Grow the per-try timeout: 200ms, 400ms, 600ms, ...
            let timeout = Duration::from_millis(200) * (attempt + 1);
            async move { call_with_timeout(timeout).await }
        })
        .when(ApiError::is_transient)
        .await
        .unwrap();
}

// ── Features: .delay_hint (honoring a server delay) ──────────────────────────────

#[derive(Debug)]
struct ThrottleError {
    retry_after: Option<Duration>,
}

impl std::fmt::Display for ThrottleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

async fn fetch() -> Result<String, ThrottleError> {
    Ok("body".into())
}

#[tokio::test]
async fn case_delay_hint() {
    let body = Retry::new()
        .max_backoff(Duration::from_secs(2))
        .attempt(fetch)
        .delay_hint(|e: &ThrottleError| e.retry_after)
        .max_delay_hint(Duration::from_secs(60))
        .await
        .unwrap();
    assert_eq!(body, "body");
}

// ── Strategies: a closure is a Policy directly ───────────────────────────────────

#[tokio::test]
async fn case_custom_policy_closure() {
    // Linear backoff: 100ms, 200ms, 300ms, ...
    let policy = |attempt: u32| Duration::from_millis(100) * (attempt + 1);
    policy.attempt(call_endpoint).max_attempts(4).await.unwrap();
}

// ── Backoff enum: exponential ────────────────────────────────────────────────────

#[tokio::test]
async fn case_exponential_backoff() {
    let _: Result<String, ApiError> = Retry::new()
        .max_attempts(4)
        .initial_backoff(Duration::from_millis(100))
        .max_backoff(Duration::from_secs(10))
        .exponential(2.0)
        .attempt(call_endpoint)
        .await;
}
