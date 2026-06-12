//! Tenacious async retries for Rust: bounded backoff with decorrelated jitter,
//! retrying until the operation succeeds or the configured budget says stop.
//!
//! Build a [`Retry`] config, call `.attempt(op)` to get an awaitable [`RetryFuture`],
//! optionally chain `.when(predicate)` to classify your error type, `.notify(fn)`
//! to observe each failed retryable attempt, `.delay_hint(fn)` to honor a
//! server-supplied delay, or `.max_delay_hint(d)` to cap how large an honored hint
//! can be. Then `.await` directly via [`IntoFuture`], or call `.run().await` to
//! drive it from a `tokio::spawn`. `Retry` is `Copy`.
//!
//! The contract is general: any `FnMut() -> impl Future<Output = Result<T, E>>` whose
//! operation is safe to re-invoke. A retry of a non-idempotent write can apply the
//! effect twice.
//!
//! # Examples
//!
//! ```
//! use terrier::Retry;
//! use std::time::Duration;
//!
//! #[derive(Debug)]
//! enum DbError { SerializationConflict, Deadlock, ConstraintViolation(String) }
//!
//! impl DbError {
//!     fn is_transient(&self) -> bool {
//!         matches!(self, Self::SerializationConflict | Self::Deadlock)
//!     }
//! }
//!
//! # async fn commit_transaction() -> Result<u64, DbError> { Ok(1) }
//! # async fn run() -> Result<u64, DbError> {
//! let rows = Retry::new()
//!     .max_attempts(5)
//!     .initial_backoff(Duration::from_millis(50))
//!     .max_backoff(Duration::from_secs(2))
//!     .attempt(commit_transaction)
//!     .when(DbError::is_transient)
//!     .notify(|err: &DbError, info| {
//!         eprintln!("attempt {} failed; retrying in {:?}", info.attempt_index, info.delay);
//!     })
//!     .await?;
//! # Ok(rows)
//! # }
//! ```

#![warn(missing_docs)]

use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::time::sleep;

// Process-wide counter mixed into the jitter source so calls landing on the
// same wall-clock tick still draw distinct fractions and decorrelate retry storms.
static JITTER_COUNTER: AtomicU64 = AtomicU64::new(0);

// Near-Duration::MAX deadlines advance the tokio timer wheel past its epoch; cap
// at 1 year so callers who set max_backoff(Duration::MAX) never hit that edge.
const SAFE_SLEEP_CAP: Duration = Duration::from_secs(365 * 24 * 3600);

/// Shared attempt-count seed for `Retry::default()` and `Budget`'s default.
pub const DEFAULT_MAX_ATTEMPTS: u32 = 3;

/// Context passed to the [`.notify`](RetryFuture::notify) callback after each
/// failed, retryable attempt, before the sleep.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub struct RetryInfo {
    /// 0-based index of the attempt that just failed.
    /// `0` is the first try, `1` the first retry, and so on.
    pub attempt_index: u32,
    /// Sleep that will follow before the next attempt.
    pub delay: Duration,
}

/// Attempt cap and optional wall-clock budget for a [`Policy`].
///
/// Returned by [`Policy::budget`]; the executor seeds its defaults from these values,
/// so a generic `P: Policy` caller sees the same caps as a direct `Retry::new()...attempt()`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    /// Total calls, first try plus retries.
    pub max_attempts: u32,
    /// Wall-clock budget across all attempts and sleeps, or `None` for no limit.
    pub max_elapsed: Option<Duration>,
}

impl Budget {
    /// Create a budget with the given attempt cap and no wall-clock limit.
    pub fn new(max_attempts: u32) -> Self {
        Self {
            max_attempts,
            max_elapsed: None,
        }
    }

    /// Set a wall-clock budget; returns `Self` for chaining.
    pub fn max_elapsed(mut self, d: Duration) -> Self {
        self.max_elapsed = Some(d);
        self
    }
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            max_attempts: DEFAULT_MAX_ATTEMPTS,
            max_elapsed: None,
        }
    }
}

// SplitMix64 finalizer: cheap, well-distributed 64-bit avalanche.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// Uniform fraction in [0, 1) from the process-wide counter mixed with wall-clock
// time, so concurrent retries landing on the same tick still decorrelate. Non-crypto.
fn jitter_fraction() -> f64 {
    let counter = JITTER_COUNTER.fetch_add(1, Ordering::Relaxed);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let time_bits = now.as_secs().rotate_left(32) ^ u64::from(now.subsec_nanos());
    let bits = splitmix64(counter ^ time_bits);
    // Fill the 52-bit f64 mantissa with the top bits of `bits`, producing a
    // standard uniform [0, 1) value: the bit pattern represents [1.0, 2.0) in
    // IEEE 754, so subtracting 1.0 yields the fraction.
    f64::from_bits(0x3FF0_0000_0000_0000 | (bits >> 12)) - 1.0
}

// Callers pass provably non-negative secs; live inputs are finite non-negative floats.
// Overflow (positive infinity from large powi) maps to SAFE_SLEEP_CAP via the clamp;
// NaN or unexpected negatives fall back to ZERO defensively.
fn clamp_to_caps(secs: f64, cap: Duration) -> Duration {
    if secs.is_nan() || secs < 0.0 {
        return Duration::ZERO;
    }
    if !secs.is_finite() {
        // Positive infinity: just return the cap (clamped to SAFE_SLEEP_CAP).
        return cap.min(SAFE_SLEEP_CAP);
    }
    Duration::try_from_secs_f64(secs)
        .unwrap_or(SAFE_SLEEP_CAP)
        .min(cap)
        .min(SAFE_SLEEP_CAP)
}

/// Internal adapter that lets a no-argument operation share the same executor path
/// as an operation that receives the attempt index.
#[derive(Debug)]
pub struct IndexIgnored<F>(F);

impl<F, Fut> IndexIgnored<F>
where
    F: FnMut() -> Fut,
{
    fn call(&mut self, _attempt: u32) -> Fut {
        (self.0)()
    }
}

/// Backoff schedule used by [`Retry`].
#[derive(Debug, Clone, Copy)]
pub(crate) enum Backoff {
    /// AWS decorrelated jitter (default): `sleep = min(max_backoff, random(initial_backoff, prev_sleep * 3))`.
    DecorrelatedJitter,
    /// Deterministic exponential: `min(max_backoff, initial_backoff * multiplier^attempt)`.
    Exponential {
        /// Growth factor. [`Retry::exponential`] panics if this is `< 1.0`.
        multiplier: f64,
    },
}

/// A per-attempt delay schedule. The sole required method is [`next_delay`](Policy::next_delay);
/// the executor ([`RetryFuture`]) owns the attempt cap, wall-clock budget, one-year sleep clamp,
/// error classification, notification, delay hints, and previous-delay state.
///
/// `next_delay(attempt, previous)` is 0-based: `next_delay(0, None)` is the sleep before
/// the *second* attempt. Returned durations may be arbitrarily large; the executor clamps them.
///
/// Override [`budget`](Policy::budget) to seed the executor's defaults from your policy's
/// own configured values (otherwise the executor starts at 3 attempts and no budget).
///
/// # Examples
///
/// ```
/// use terrier::{Fixed, Policy};
/// use std::time::Duration;
///
/// # async fn example() -> Result<u8, String> {
/// let out = Fixed(Duration::from_millis(10))
///     .attempt(|| async { Ok::<u8, String>(7) })
///     .max_attempts(4)
///     .await?;
/// # Ok(out)
/// # }
/// ```
pub trait Policy: Sized {
    /// Sleep before the next attempt. `attempt` is 0-based; `previous` is the delay
    /// actually slept (post-clamp, after any hint) before the most recent retry.
    ///
    /// Stateless policies may ignore `previous`. Stateful schedules that need it
    /// (e.g. decorrelated jitter) use it to widen the next draw's window.
    fn next_delay(&self, attempt: u32, previous: Option<Duration>) -> Duration;

    /// Return this policy's configured attempt cap and optional elapsed budget.
    ///
    /// The executor seeds its defaults from these values, so a generic `P: Policy`
    /// caller gets the same caps as a direct `Retry::new()...attempt()` call.
    fn budget(&self) -> Budget {
        Budget::default()
    }

    /// Attach an operation, returning an awaitable [`RetryFuture`] seeded from [`budget`](Policy::budget).
    fn attempt<T, E, F, Fut>(self, op: F) -> RetryFuture<T, E, IndexIgnored<F>, Fut, Self>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T, E>>,
    {
        let b = self.budget();
        RetryFuture::new_seeded(self, IndexIgnored(op), b.max_attempts, b.max_elapsed, None)
    }

    /// Like [`attempt`](Policy::attempt) but passes the 0-based attempt index to the
    /// operation: `0` is the first try, `1` the first retry, and so on.
    fn attempt_with<T, E, F, Fut>(self, op: F) -> RetryFuture<T, E, F, Fut, Self>
    where
        F: FnMut(u32) -> Fut,
        Fut: Future<Output = Result<T, E>>,
    {
        let b = self.budget();
        RetryFuture::new_seeded(self, op, b.max_attempts, b.max_elapsed, None)
    }
}

/// Flat-delay policy: every sleep is `d` regardless of attempt count.
///
/// Useful for cases where the server prescribes a fixed retry interval, or in
/// tests where a deterministic, predictable schedule matters more than backoff growth.
///
/// # Examples
///
/// ```
/// use terrier::{Fixed, Policy};
/// use std::time::Duration;
///
/// # async fn example() -> Result<(), String> {
/// Fixed(Duration::from_millis(100))
///     .attempt(|| async { Ok::<(), String>(()) })
///     .max_attempts(3)
///     .await?;
/// # Ok(())
/// # }
/// ```
#[must_use]
#[derive(Debug, Clone, Copy)]
pub struct Fixed(pub Duration);

impl Policy for Fixed {
    fn next_delay(&self, _attempt: u32, _previous: Option<Duration>) -> Duration {
        self.0
    }
}

/// Decorrelated-jitter policy.
///
/// The retry executor uses the AWS recurrence: `sleep = min(max_backoff,
/// random(base, previous_sleep * 3))`, with the first sleep seeded from `base`.
///
/// # Examples
///
/// ```
/// use terrier::{DecorrelatedJitter, Policy};
/// use std::time::Duration;
///
/// # async fn call() -> Result<(), String> { Ok(()) }
/// # async fn example() -> Result<(), String> {
/// DecorrelatedJitter::new(Duration::from_millis(100), Duration::from_secs(20))
///     .attempt(call)
///     .max_attempts(6)
///     .await?;
/// # Ok(())
/// # }
/// ```
#[must_use]
#[derive(Debug, Clone, Copy)]
pub struct DecorrelatedJitter {
    base: Duration,
    max_backoff: Duration,
}

impl DecorrelatedJitter {
    /// Decorrelated jitter from `base` up to `max_backoff`.
    pub fn new(base: Duration, max_backoff: Duration) -> Self {
        Self { base, max_backoff }
    }

    // AWS decorrelated jitter with a caller-supplied fraction in [0,1).
    // Preserves the AWS oracle test seam: uniform(a,b) = a + f*(b-a).
    fn sample_with(&self, previous: Option<Duration>, fraction: f64) -> Duration {
        let previous = previous.unwrap_or(self.base);
        let base_f = self.base.as_secs_f64();
        let upper_f = (previous.as_secs_f64() * 3.0).max(base_f);
        let secs = base_f + fraction * (upper_f - base_f);
        clamp_to_caps(secs, self.max_backoff)
    }

    fn sample(&self, previous: Option<Duration>) -> Duration {
        self.sample_with(previous, jitter_fraction())
    }
}

impl Policy for DecorrelatedJitter {
    fn next_delay(&self, _attempt: u32, previous: Option<Duration>) -> Duration {
        self.sample(previous)
    }
}

/// Implements `Policy` for any closure `F: Fn(u32) -> Duration`, treating the closure
/// as a stateless delay schedule. The attempt index is 0-based; `previous` is ignored.
///
/// # Examples
///
/// ```
/// use terrier::Policy;
/// use std::time::Duration;
///
/// # async fn example() -> Result<(), String> {
/// // Linear backoff: 100ms, 200ms, 300ms, ...
/// let policy = |attempt: u32| Duration::from_millis(100) * (attempt + 1);
/// policy.attempt(|| async { Ok::<(), String>(()) })
///     .max_attempts(3)
///     .await?;
/// # Ok(())
/// # }
/// ```
impl<F: Fn(u32) -> Duration> Policy for F {
    fn next_delay(&self, attempt: u32, _previous: Option<Duration>) -> Duration {
        self(attempt)
    }
}

/// Retry policy and default backoff schedule. `Copy` - pass by value without allocation.
///
/// Defaults: 3 attempts, 500 ms initial backoff, 30 s cap, decorrelated jitter.
///
/// # Examples
///
/// ```
/// use terrier::Retry;
/// use std::time::Duration;
///
/// #[derive(Debug)]
/// enum MyError { Transient, Fatal }
/// impl MyError { fn is_transient(&self) -> bool { matches!(self, Self::Transient) } }
///
/// # async fn op() -> Result<String, MyError> { Ok("ok".into()) }
/// # async fn example() -> Result<String, MyError> {
/// let out = Retry::new()
///     .max_attempts(5)
///     .initial_backoff(Duration::from_millis(200))
///     .max_backoff(Duration::from_secs(10))
///     .attempt(op)
///     .when(MyError::is_transient)
///     .await?;
/// # Ok(out)
/// # }
/// ```
#[must_use]
#[derive(Debug, Clone, Copy)]
pub struct Retry {
    max_attempts: u32,
    initial_backoff: Duration,
    max_backoff: Duration,
    backoff: Backoff,
    max_elapsed: Option<Duration>,
}

impl Default for Retry {
    fn default() -> Self {
        Self {
            max_attempts: DEFAULT_MAX_ATTEMPTS,
            initial_backoff: Duration::from_millis(500),
            max_backoff: Duration::from_secs(30),
            backoff: Backoff::DecorrelatedJitter,
            max_elapsed: None,
        }
    }
}

impl Retry {
    /// Create a config with the default knobs (3 attempts, 500 ms, 30 s cap,
    /// decorrelated jitter).
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Total calls (first try + retries). Panics if `n` is 0.
    ///
    /// # Panics
    ///
    /// Panics if `n` is 0.
    #[inline]
    pub fn max_attempts(mut self, n: u32) -> Self {
        assert!(n >= 1, "max_attempts must be at least 1");
        self.max_attempts = n;
        self
    }

    /// Backoff applied before the second attempt.
    #[inline]
    pub fn initial_backoff(mut self, d: Duration) -> Self {
        self.initial_backoff = d;
        self
    }

    /// Maximum single sleep regardless of how large the computed backoff grows.
    #[inline]
    pub fn max_backoff(mut self, d: Duration) -> Self {
        self.max_backoff = d;
        self
    }

    /// Use AWS decorrelated-jitter backoff (the default).
    ///
    /// Each sleep is drawn as `min(max_backoff, random(initial_backoff, prev_sleep * 3))`.
    #[inline]
    pub fn decorrelated_jitter(mut self) -> Self {
        self.backoff = Backoff::DecorrelatedJitter;
        self
    }

    /// Use deterministic exponential backoff: `min(max_backoff, initial_backoff * multiplier^attempt)`.
    ///
    /// # Panics
    ///
    /// Panics if `multiplier < 1.0`.
    #[inline]
    pub fn exponential(mut self, multiplier: f64) -> Self {
        assert!(multiplier >= 1.0, "multiplier must be >= 1.0");
        self.backoff = Backoff::Exponential { multiplier };
        self
    }

    /// Hard wall-clock budget across all attempts including sleeps. Checked before each
    /// sleep: if `elapsed + next_delay >= budget` the last error is returned immediately,
    /// without sleeping, even if attempts remain.
    #[inline]
    pub fn max_elapsed(mut self, d: Duration) -> Self {
        self.max_elapsed = Some(d);
        self
    }

    /// Compute the sleep for `attempt` (0-based) using the configured schedule.
    ///
    /// With [`Backoff::DecorrelatedJitter`], each draw is a fresh independent sample
    /// (not the correlated executor sequence). With [`Backoff::Exponential`], returns
    /// the deterministic `min(max_backoff, initial_backoff * multiplier^attempt)`.
    /// Results are clamped to 1 year so `max_backoff(Duration::MAX)` is safe.
    ///
    /// With jittered backoff this is an independent sample and does not predict
    /// the sequence the executor will actually sleep.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn delay_for_attempt(&self, attempt: u32) -> Duration {
        match self.backoff {
            Backoff::Exponential { multiplier } => {
                exponential_delay(self.initial_backoff, self.max_backoff, multiplier, attempt)
            }
            Backoff::DecorrelatedJitter => {
                DecorrelatedJitter::new(self.initial_backoff, self.max_backoff).sample(None)
            }
        }
    }

    /// Attach an operation, returning an awaitable [`RetryFuture`] seeded from this config.
    ///
    /// Chain `.when(predicate)` to restrict which errors are retried;
    /// without `.when`, all errors are retried up to `max_attempts`.
    ///
    /// # Examples
    ///
    /// ```
    /// use terrier::Retry;
    /// use std::time::Duration;
    ///
    /// # async fn example() -> Result<String, String> {
    /// let out = Retry::new()
    ///     .max_attempts(3)
    ///     .initial_backoff(Duration::from_millis(100))
    ///     .exponential(2.0)
    ///     .attempt(|| async { Ok::<String, String>("ok".into()) })
    ///     .await?;
    /// # Ok(out)
    /// # }
    /// ```
    pub fn attempt<T, E, F, Fut>(self, op: F) -> RetryFuture<T, E, IndexIgnored<F>, Fut, Retry>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T, E>>,
    {
        <Retry as Policy>::attempt(self, op)
    }

    /// Attach an operation that receives the 0-based attempt index, returning an
    /// awaitable [`RetryFuture`] seeded from this config. `0` is the first try,
    /// `1` the first retry, and so on.
    ///
    /// Use this when the call itself must vary per attempt: an escalating per-try
    /// timeout, a widening page size, a rotating endpoint. For observation only
    /// (logging, metrics), prefer [`notify`](RetryFuture::notify).
    ///
    /// # Examples
    ///
    /// ```
    /// use terrier::Retry;
    /// use std::time::Duration;
    ///
    /// # async fn call_with_timeout(_t: Duration) -> Result<u8, String> { Ok(1) }
    /// # async fn example() -> Result<u8, String> {
    /// let out = Retry::new()
    ///     .initial_backoff(Duration::from_millis(50))
    ///     .attempt_with(|attempt| {
    ///         let timeout = Duration::from_millis(200) * (attempt + 1);
    ///         async move { call_with_timeout(timeout).await }
    ///     })
    ///     .await?;
    /// # Ok(out)
    /// # }
    /// ```
    pub fn attempt_with<T, E, F, Fut>(self, op: F) -> RetryFuture<T, E, F, Fut, Retry>
    where
        F: FnMut(u32) -> Fut,
        Fut: Future<Output = Result<T, E>>,
    {
        <Retry as Policy>::attempt_with(self, op)
    }
}

impl Policy for Retry {
    fn next_delay(&self, attempt: u32, previous: Option<Duration>) -> Duration {
        match self.backoff {
            Backoff::DecorrelatedJitter => {
                DecorrelatedJitter::new(self.initial_backoff, self.max_backoff).sample(previous)
            }
            Backoff::Exponential { multiplier } => {
                exponential_delay(self.initial_backoff, self.max_backoff, multiplier, attempt)
            }
        }
    }

    fn budget(&self) -> Budget {
        Budget {
            max_attempts: self.max_attempts,
            max_elapsed: self.max_elapsed,
        }
    }

    fn attempt<T, E, F, Fut>(self, op: F) -> RetryFuture<T, E, IndexIgnored<F>, Fut, Self>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T, E>>,
    {
        let b = self.budget();
        RetryFuture::new_seeded(
            self,
            IndexIgnored(op),
            b.max_attempts,
            b.max_elapsed,
            Some(self.max_backoff),
        )
    }

    fn attempt_with<T, E, F, Fut>(self, op: F) -> RetryFuture<T, E, F, Fut, Self>
    where
        F: FnMut(u32) -> Fut,
        Fut: Future<Output = Result<T, E>>,
    {
        let b = self.budget();
        RetryFuture::new_seeded(
            self,
            op,
            b.max_attempts,
            b.max_elapsed,
            Some(self.max_backoff),
        )
    }
}

// O(1) closed-form deterministic capped exponential delay.
// Handles base==0 (returns ZERO before the powi to avoid 0*inf=NaN) and
// overflow/inf from large attempt*multiplier (clamp_to_caps handles non-finite).
fn exponential_delay(base: Duration, cap: Duration, growth: f64, attempt: u32) -> Duration {
    let base_f = base.as_secs_f64();
    if base_f == 0.0 {
        return Duration::ZERO;
    }
    // Cap the exponent to i32::MAX to avoid powi overflow on u32::MAX attempts.
    let exp = attempt.min(i32::MAX as u32) as i32;
    let secs = base_f * growth.powi(exp);
    clamp_to_caps(secs, cap)
}

/// Awaitable produced by [`Retry::attempt`] and [`Policy::attempt`].
///
/// Awaitable via [`IntoFuture`] or by calling [`run()`](RetryFuture::run) directly.
/// Use `.run()` when you need to `tokio::spawn` the retry loop
/// (`tokio::spawn(retry.run())` works; spawning the `IntoFuture` directly does not
/// because it lacks `Send + 'static` bounds on the captured state).
///
/// Chain `.when(|e| ...)` to restrict retries to transient errors, `.notify(fn)` to
/// observe each failed retryable attempt, `.delay_hint(|e| ...)` to honor a
/// server-supplied delay, and `.max_delay_hint(d)` to cap how large an honored hint
/// can be. For custom policies, `.max_attempts(n)` / `.max_elapsed(d)` set the
/// executor's budget here.
///
/// # `IntoFuture` bounds
///
/// `.await` requires `T: Send + 'static`, `E: Send + 'static`,
/// `F: Send + 'static`, `Fut: Send + 'static`, `P: Send + 'static`.
/// Use a `move` closure to satisfy the `'static` bound when capturing local state.
#[must_use = "a RetryFuture does nothing unless awaited or run() is called"]
pub struct RetryFuture<T, E, F, Fut, P = Retry> {
    policy: P,
    op: F,
    max_attempts: u32,
    max_elapsed: Option<Duration>,
    // None means retry all errors.
    #[allow(clippy::type_complexity)]
    predicate: Option<Box<dyn FnMut(&E) -> bool + Send>>,
    // Each hook boxes once on registration; per-attempt cost is the call, not allocation.
    #[allow(clippy::type_complexity)]
    notifier: Option<Box<dyn FnMut(&E, RetryInfo) + Send>>,
    #[allow(clippy::type_complexity)]
    delay_hint: Option<Box<dyn FnMut(&E) -> Option<Duration> + Send>>,
    // Ceiling on an honored hint. None means no policy-level cap (only SAFE_SLEEP_CAP applies).
    max_delay_hint: Option<Duration>,
    _marker: std::marker::PhantomData<fn() -> (T, Fut)>,
}

mod sealed {
    pub trait Sealed {}
}

/// Adapter trait used by the executor for both `.attempt` and `.attempt_with`.
///
/// Sealed: not part of the stable API surface; implement [`Policy`] instead.
pub trait CallWithIndex<Fut>: sealed::Sealed {
    #[doc(hidden)]
    fn call_indexed(&mut self, attempt: u32) -> Fut;
}

impl<F, Fut> sealed::Sealed for IndexIgnored<F> where F: FnMut() -> Fut {}
impl<F, Fut> CallWithIndex<Fut> for IndexIgnored<F>
where
    F: FnMut() -> Fut,
{
    fn call_indexed(&mut self, attempt: u32) -> Fut {
        self.call(attempt)
    }
}

impl<F, Fut> sealed::Sealed for F where F: FnMut(u32) -> Fut {}
impl<F, Fut> CallWithIndex<Fut> for F
where
    F: FnMut(u32) -> Fut,
{
    fn call_indexed(&mut self, attempt: u32) -> Fut {
        self(attempt)
    }
}

impl<T, E, F, Fut, P> RetryFuture<T, E, F, Fut, P>
where
    F: CallWithIndex<Fut>,
    Fut: Future<Output = Result<T, E>>,
    P: Policy,
{
    fn new_seeded(
        policy: P,
        op: F,
        max_attempts: u32,
        max_elapsed: Option<Duration>,
        max_delay_hint: Option<Duration>,
    ) -> Self {
        Self {
            policy,
            op,
            max_attempts,
            max_elapsed,
            predicate: None,
            notifier: None,
            delay_hint: None,
            max_delay_hint,
            _marker: std::marker::PhantomData,
        }
    }

    /// Total calls (first try + retries). Panics if `n` is 0.
    ///
    /// [`Retry::attempt`] seeds this from the `Retry` config; set it here for a custom
    /// [`Policy`], which defaults to 3.
    ///
    /// # Panics
    ///
    /// Panics if `n` is 0.
    pub fn max_attempts(mut self, n: u32) -> Self {
        assert!(n >= 1, "max_attempts must be at least 1");
        self.max_attempts = n;
        self
    }

    /// Hard wall-clock budget across all attempts including sleeps. Checked before each
    /// sleep: if `elapsed + next_delay >= budget` the last error is returned immediately,
    /// without sleeping, even if attempts remain.
    pub fn max_elapsed(mut self, d: Duration) -> Self {
        self.max_elapsed = Some(d);
        self
    }

    /// Only retry errors for which `predicate` returns `true`.
    /// Without this, all errors are retried up to `max_attempts`.
    pub fn when<W>(mut self, predicate: W) -> Self
    where
        W: FnMut(&E) -> bool + Send + 'static,
    {
        self.predicate = Some(Box::new(predicate));
        self
    }

    /// Register a callback invoked after each failed, retryable attempt - after `.when`
    /// accepts the error and after the budget pre-check passes, so it fires exactly when
    /// a sleep will follow.
    ///
    /// Not called on first-try success, on the final exhausted attempt, when `.when`
    /// rejects the error, or when the budget pre-check stops early.
    pub fn notify<N>(mut self, n: N) -> Self
    where
        N: FnMut(&E, RetryInfo) + Send + 'static,
    {
        self.notifier = Some(Box::new(n));
        self
    }

    /// Extract a server-supplied delay (e.g. an HTTP `Retry-After`) from the error.
    ///
    /// When the closure returns `Some(d)` for a retryable error, `d` replaces the
    /// policy's computed delay for that attempt. The honored hint is first clamped to
    /// [`max_delay_hint`](RetryFuture::max_delay_hint) (default: `max_backoff` for
    /// `Retry`; the one-year cap for custom policies), then the budget pre-check and
    /// the one-year sleep clamp apply as usual. The hinted (post-clamp) sleep is fed
    /// back as `previous` to the next policy draw, widening the next
    /// decorrelated-jitter window.
    pub fn delay_hint<H>(mut self, h: H) -> Self
    where
        H: FnMut(&E) -> Option<Duration> + Send + 'static,
    {
        self.delay_hint = Some(Box::new(h));
        self
    }

    /// Cap the delay that an honored server hint may produce.
    ///
    /// When `.delay_hint` returns `Some(d)`, the effective sleep is
    /// `min(d, max_delay_hint)` before the budget pre-check and one-year clamp.
    /// `Retry::attempt` seeds this to `max_backoff` by default, so a garbage
    /// `Retry-After` (e.g. an epoch timestamp parsed as seconds) is bounded without
    /// any extra configuration. Raise it — e.g. `.max_delay_hint(Duration::from_secs(300))`
    /// — to honor a longer server delay while keeping a short policy backoff ceiling.
    /// A custom `Policy` (e.g. `Fixed`, a closure) seeds this as `None`, leaving the
    /// hint bounded only by the one-year cap; set it explicitly if you need a tighter
    /// bound.
    pub fn max_delay_hint(mut self, d: Duration) -> Self {
        self.max_delay_hint = Some(d);
        self
    }

    /// Drive the retry loop to completion.
    ///
    /// This is the same future that `.await` runs via [`IntoFuture`]; calling it
    /// directly lets you `tokio::spawn(retry.run())` when `T`, `E`, `F`, `Fut`,
    /// and `P` are all `Send + 'static`.
    pub async fn run(mut self) -> Result<T, E> {
        let mut is_retryable = self.predicate;
        let mut notifier = self.notifier;
        let mut delay_hint = self.delay_hint;
        let start = self.max_elapsed.map(|_| tokio::time::Instant::now());
        let mut previous_delay = None;

        for attempt in 0..self.max_attempts {
            match self.op.call_indexed(attempt).await {
                Ok(value) => return Ok(value),
                Err(err) => {
                    let remaining = self.max_attempts - attempt - 1;

                    if remaining == 0 {
                        return Err(err);
                    }

                    if let Some(ref mut pred) = is_retryable {
                        if !pred(&err) {
                            return Err(err);
                        }
                    }

                    // hint > policy delay; when a hint is honored it is clamped to
                    // max_delay_hint first, then SAFE_SLEEP_CAP always applies.
                    let raw = delay_hint
                        .as_mut()
                        .and_then(|h| h(&err))
                        .map(|hint| {
                            let cap = self.max_delay_hint.unwrap_or(SAFE_SLEEP_CAP);
                            hint.min(cap)
                        })
                        .unwrap_or_else(|| self.policy.next_delay(attempt, previous_delay));
                    let delay = raw.min(SAFE_SLEEP_CAP);

                    // Pre-check: if elapsed + next sleep already exceeds the budget,
                    // return now rather than sleeping only to bail immediately after.
                    if let (Some(budget), Some(started)) = (self.max_elapsed, start) {
                        if started.elapsed().saturating_add(delay) >= budget {
                            return Err(err);
                        }
                    }

                    if let Some(ref mut notify) = notifier {
                        notify(
                            &err,
                            RetryInfo {
                                attempt_index: attempt,
                                delay,
                            },
                        );
                    }

                    sleep(delay).await;
                    previous_delay = Some(delay);
                }
            }
        }

        unreachable!("max_attempts >= 1 guarantees the loop returns")
    }
}

impl<T, E, F, Fut, P> IntoFuture for RetryFuture<T, E, F, Fut, P>
where
    T: Send + 'static,
    E: Send + 'static,
    F: CallWithIndex<Fut> + Send + 'static,
    Fut: Future<Output = Result<T, E>> + Send + 'static,
    P: Policy + Send + 'static,
{
    type Output = Result<T, E>;
    type IntoFuture = Pin<Box<dyn Future<Output = Result<T, E>> + Send>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.run())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::future::IntoFuture as _;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::{Duration, Instant};

    use super::*;

    // ── schedule computation (no I/O) ────────────────────────────────────────

    #[test]
    fn delay_exponential() {
        let cfg = Retry::new()
            .initial_backoff(Duration::from_millis(100))
            .exponential(2.0)
            .max_backoff(Duration::from_secs(60));
        assert_eq!(cfg.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(cfg.delay_for_attempt(1), Duration::from_millis(200));
        assert_eq!(cfg.delay_for_attempt(2), Duration::from_millis(400));
    }

    #[test]
    fn delay_respects_max_backoff() {
        let cfg = Retry::new()
            .initial_backoff(Duration::from_secs(10))
            .max_backoff(Duration::from_secs(5))
            .exponential(4.0);
        assert_eq!(cfg.delay_for_attempt(2), Duration::from_secs(5));
    }

    #[test]
    fn delay_caps_on_huge_attempt() {
        assert_eq!(
            exponential_delay(
                Duration::from_millis(500),
                Duration::from_secs(30),
                2.0,
                100
            ),
            Duration::from_secs(30)
        );
    }

    #[test]
    fn delay_caps_with_large_multiplier() {
        assert_eq!(
            exponential_delay(
                Duration::from_millis(500),
                Duration::from_secs(30),
                100.0,
                20
            ),
            Duration::from_secs(30)
        );
    }

    // BLOCK regression: exponential_delay(u32::MAX) with multiplier(1.0+EPSILON) must
    // return in well under a second — the old loop would spin ~4.29B iterations.
    #[test]
    fn exponential_delay_u32_max_attempt_returns_fast() {
        let start = Instant::now();
        let d = exponential_delay(
            Duration::from_nanos(1),
            Duration::from_secs(30),
            1.0 + f64::EPSILON,
            u32::MAX,
        );
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_millis(100),
            "exponential_delay with u32::MAX attempt took {:?} (must be <100ms)",
            elapsed
        );
        // With multiplier barely above 1.0 and base 1ns, this grows to cap quickly.
        assert!(d <= Duration::from_secs(30));
    }

    // First decorrelated-jitter draw lies in [base, base*3] and varies.
    #[test]
    fn jitter_stays_within_decorrelated_bounds_and_varies() {
        let base = Duration::from_secs(1);
        let max_backoff = Duration::from_secs(30);
        let policy = DecorrelatedJitter::new(base, max_backoff);
        // attempt=0: prev=1s, upper=min(30s, 3s)=3s, so draws are in [1s, 3s].
        let upper = Duration::from_secs(3);
        let mut seen: HashSet<u128> = HashSet::new();
        let mut any_above_base = false;
        for _ in 0..1000 {
            let d = policy.sample(None);
            assert!(d >= base, "jitter must never go below base");
            assert!(d <= upper, "jitter must never exceed prev*3 bound");
            seen.insert(d.as_nanos());
            if d > base {
                any_above_base = true;
            }
        }
        assert!(
            seen.len() > 1,
            "jitter must produce more than one distinct value"
        );
        assert!(any_above_base, "jitter must occasionally draw above base");
    }

    // exponential schedule: prev = min(cap, base * mult^attempt).
    #[test]
    fn default_schedule_exponential_is_deterministic() {
        let base = Duration::from_millis(100);
        let cap = Duration::from_secs(10);
        // 100ms * 2^0..6 = 100, 200, 400, 800, 1600, 3200, 6400 ms; all below 10s cap.
        assert_eq!(
            exponential_delay(base, cap, 2.0, 0),
            Duration::from_millis(100)
        );
        assert_eq!(
            exponential_delay(base, cap, 2.0, 1),
            Duration::from_millis(200)
        );
        assert_eq!(
            exponential_delay(base, cap, 2.0, 2),
            Duration::from_millis(400)
        );
        assert_eq!(
            exponential_delay(base, cap, 2.0, 6),
            Duration::from_millis(6400)
        );
        // attempt=7: 100ms * 2^7 = 12800ms > 10s, so it is clamped to cap.
        assert_eq!(exponential_delay(base, cap, 2.0, 7), cap);
    }

    // jitter(decorrelated) draws in [base, min(cap, previous_sleep*3)] for each retry.
    #[test]
    fn default_schedule_jitter_uses_previous_sleep_bounds() {
        let base = Duration::from_millis(100);
        let cap = Duration::from_secs(20);
        let policy = DecorrelatedJitter::new(base, cap);

        for _ in 0..200 {
            let mut previous = None;
            for _ in 0..8u32 {
                let previous_for_bound = previous.unwrap_or(base);
                let upper = (previous_for_bound * 3).min(cap);
                let d = policy.sample(previous);
                assert!(d >= base, "{d:?} below base {base:?}");
                assert!(d <= upper, "{d:?} above prev*3 bound {upper:?}");
                previous = Some(d);
            }
        }
    }

    #[test]
    fn decorrelated_jitter_caps_after_random_draw() {
        let base = Duration::from_secs(1);
        let cap = Duration::from_secs(2);
        let policy = DecorrelatedJitter::new(base, cap);

        let d = policy.sample_with(Some(base), 0.75);

        assert_eq!(d, cap, "1s + 0.75 * (3s - 1s) = 2.5s, then the cap applies");
    }

    // base > previous*3: cap < base edge case.
    #[test]
    fn decorrelated_jitter_base_greater_than_prev_times_three() {
        // When base > previous*3, upper_f is clamped to base_f (the .max(base_f) guard),
        // so the draw collapses to exactly base regardless of fraction.
        let base = Duration::from_secs(10);
        let cap = Duration::from_secs(60);
        let policy = DecorrelatedJitter::new(base, cap);
        let prev = Duration::from_millis(100); // prev*3 = 300ms < base = 10s
        let d0 = policy.sample_with(Some(prev), 0.0);
        let d1 = policy.sample_with(Some(prev), 1.0);
        assert_eq!(d0, base, "lower bound is always base");
        assert_eq!(
            d1, base,
            "upper bound also collapses to base when prev*3 < base"
        );
    }

    // initial_backoff(ZERO) + decorrelated jitter: every draw is 0ns (0 upper bound).
    #[test]
    fn decorrelated_jitter_zero_base_produces_zero_stream() {
        // With base=0, upper=max(0*3, 0)=0, so secs = 0 + f*(0-0) = 0 always.
        let policy = DecorrelatedJitter::new(Duration::ZERO, Duration::from_secs(60));
        for _ in 0..100 {
            assert_eq!(
                policy.sample(None),
                Duration::ZERO,
                "zero base must produce zero delay"
            );
        }
    }

    // ── AWS reference parity ─────────────────────────────────────────────────
    // Oracle ported from aws-samples/aws-arch-backoff-simulator (ExpoBackoffDecorr):
    //   self.sleep = self.base
    //   self.sleep = min(self.cap, random.uniform(self.base, self.sleep * 3))
    // with the uniform draw's fraction injected: uniform(a, b) = a + f*(b - a).
    fn aws_decorr_oracle_step(base: f64, cap: f64, prev_sleep: f64, fraction: f64) -> f64 {
        (base + fraction * (prev_sleep * 3.0 - base)).min(cap)
    }

    /// Each chained draw matches the AWS reference recurrence when fed the same
    /// fraction, threading terrier's own (nanosecond-rounded) previous sleep.
    #[test]
    fn aws_oracle_recurrence_parity() {
        let base = Duration::from_millis(100);
        let cap = Duration::from_secs(10);
        let base_f = base.as_secs_f64();
        let cap_f = cap.as_secs_f64();
        let policy = DecorrelatedJitter::new(base, cap);
        // Spans low/mid/high fractions; the 0.999 runs drive the sleep up to the cap,
        // exercising cap-after-draw and the capped value feeding the next bound.
        let fractions = [
            0.0, 0.999, 0.5, 0.25, 0.875, 0.999, 0.999, 0.1, 0.999, 0.999, 0.999, 0.0, 0.6,
        ];
        let mut previous: Option<Duration> = None;
        for (i, &f) in fractions.iter().enumerate() {
            let prev_f = previous.unwrap_or(base).as_secs_f64();
            let expected = aws_decorr_oracle_step(base_f, cap_f, prev_f, f);
            let d = policy.sample_with(previous, f);
            let got = d.as_secs_f64();
            assert!(
                (got - expected).abs() < 1e-8,
                "step {i}: terrier {got} vs AWS oracle {expected}"
            );
            previous = Some(d);
        }
    }

    /// Seeded at base, and the upper bound tracks the actual fed-back sleep
    /// (true statefulness) while the lower bound stays at base, per the reference
    /// `uniform(base, sleep * 3)`.
    #[test]
    fn decorrelated_jitter_seeded_at_base_and_stateful_bounds() {
        let base = Duration::from_millis(100);
        let cap = Duration::from_secs(60);
        let policy = DecorrelatedJitter::new(base, cap);

        // First retry (no previous sleep): draw spans [base, base*3].
        assert_eq!(policy.sample_with(None, 0.0), base);
        assert_eq!(policy.sample_with(None, 1.0), base * 3);

        // A larger previous sleep widens the upper bound...
        let prev = Duration::from_secs(10);
        assert_eq!(policy.sample_with(Some(prev), 1.0), Duration::from_secs(30));
        // ...while the lower bound stays at base, not at the previous sleep.
        assert_eq!(policy.sample_with(Some(prev), 0.0), base);
    }

    /// End-to-end through the executor: with jitter on, every observed sleep lies in
    /// [base, min(cap, previous_actual_sleep * 3)] — the executor feeds back the
    /// sampled sleep, not a reconstructed deterministic schedule.
    #[tokio::test(start_paused = true)]
    async fn executor_threads_actual_sampled_sleep_within_aws_bounds() {
        let base = Duration::from_millis(100);
        let cap = Duration::from_secs(10);
        for _ in 0..50 {
            let delays = Arc::new(std::sync::Mutex::new(Vec::<Duration>::new()));
            let d = Arc::clone(&delays);
            let _: Result<(), String> = Retry::new()
                .max_attempts(8)
                .initial_backoff(base)
                .max_backoff(cap)
                .attempt(|| async { Err::<(), _>("transient".into()) })
                .notify(move |_e, info| d.lock().unwrap().push(info.delay))
                .await;
            let got = delays.lock().unwrap().clone();
            assert_eq!(got.len(), 7, "8 attempts give 7 sleeps");
            let mut prev = base;
            for (i, &delay) in got.iter().enumerate() {
                let upper = (prev * 3).min(cap);
                assert!(delay >= base, "sleep {i}: {delay:?} below base {base:?}");
                assert!(
                    delay <= upper,
                    "sleep {i}: {delay:?} above min(cap, prev*3) = {upper:?}"
                );
                prev = delay;
            }
        }
    }

    // ── jitter_fraction distribution ─────────────────────────────────────────

    #[test]
    fn jitter_fraction_stays_in_range() {
        for _ in 0..10_000 {
            let f = jitter_fraction();
            assert!((0.0..1.0).contains(&f), "jitter_fraction {f} out of [0,1)");
        }
    }

    #[test]
    fn jitter_fraction_multi_thread_distinctness() {
        use std::sync::Mutex;
        let samples = Arc::new(Mutex::new(Vec::<u64>::new()));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let s = Arc::clone(&samples);
                std::thread::spawn(move || {
                    for _ in 0..64 {
                        let f = jitter_fraction();
                        // Store raw bits for exact comparison.
                        s.lock().unwrap().push(f.to_bits());
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        let v = samples.lock().unwrap().clone();
        let unique: HashSet<u64> = v.iter().copied().collect();
        // 512 draws across 8 threads: expect high uniqueness (collisions are astronomically rare).
        assert!(
            unique.len() > 400,
            "expected >400 unique fractions from 512 draws, got {}",
            unique.len()
        );
    }

    // ── retry loop correctness ────────────────────────────────────────────────

    #[tokio::test]
    async fn succeeds_on_first_attempt() {
        let calls = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&calls);
        let result: Result<&str, String> = Retry::new()
            .exponential(2.0)
            .attempt(move || {
                c.fetch_add(1, Ordering::Relaxed);
                async { Ok("ok") }
            })
            .await;
        assert_eq!(result.unwrap(), "ok");
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    /// max_attempts=1 means no retries; the single failure is returned immediately.
    #[tokio::test]
    async fn max_attempts_one_no_retry() {
        let calls = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&calls);
        let result: Result<&str, String> = Retry::new()
            .max_attempts(1)
            .initial_backoff(Duration::from_millis(1))
            .exponential(2.0)
            .attempt(move || {
                c.fetch_add(1, Ordering::Relaxed);
                async { Err::<&str, _>("single shot".into()) }
            })
            .await;
        assert_eq!(result.unwrap_err(), "single shot");
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn retries_then_succeeds() {
        let calls = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&calls);
        let result: Result<&str, String> = Retry::new()
            .max_attempts(3)
            .initial_backoff(Duration::from_millis(1))
            .exponential(2.0)
            .attempt(move || {
                let n = c.fetch_add(1, Ordering::Relaxed);
                async move {
                    if n < 2 {
                        Err("transient".into())
                    } else {
                        Ok("recovered")
                    }
                }
            })
            .await;
        assert_eq!(result.unwrap(), "recovered");
        assert_eq!(calls.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn exhausts_all_attempts() {
        let calls = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&calls);
        let result: Result<&str, String> = Retry::new()
            .max_attempts(3)
            .initial_backoff(Duration::from_millis(1))
            .exponential(2.0)
            .attempt(move || {
                c.fetch_add(1, Ordering::Relaxed);
                async { Err::<&str, _>("always fails".into()) }
            })
            .await;
        assert_eq!(result.unwrap_err(), "always fails");
        assert_eq!(calls.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn stops_on_non_retryable() {
        let calls = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&calls);
        let result: Result<&str, String> = Retry::new()
            .max_attempts(5)
            .initial_backoff(Duration::from_millis(1))
            .exponential(2.0)
            .attempt(move || {
                c.fetch_add(1, Ordering::Relaxed);
                async { Err::<&str, _>("fatal error".into()) }
            })
            .when(|e: &String| !e.contains("fatal"))
            .await;
        assert_eq!(result.unwrap_err(), "fatal error");
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn no_when_retries_all_errors() {
        let calls = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&calls);
        let result: Result<&str, String> = Retry::new()
            .max_attempts(3)
            .initial_backoff(Duration::from_millis(1))
            .exponential(2.0)
            .attempt(move || {
                c.fetch_add(1, Ordering::Relaxed);
                async { Err::<&str, _>("any error".into()) }
            })
            .await;
        assert!(result.is_err());
        assert_eq!(
            calls.load(Ordering::Relaxed),
            3,
            "without .when, all errors are retried"
        );
    }

    #[tokio::test]
    async fn when_predicate_receives_correct_error_value() {
        let calls = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&calls);
        let seen_values = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let sv = Arc::clone(&seen_values);
        let result: Result<&str, String> = Retry::new()
            .max_attempts(3)
            .initial_backoff(Duration::from_millis(1))
            .exponential(2.0)
            .attempt(move || {
                c.fetch_add(1, Ordering::Relaxed);
                async { Err::<&str, _>("transient".into()) }
            })
            .when(move |e: &String| {
                sv.lock().unwrap().push(e.clone());
                e.contains("transient")
            })
            .await;
        assert!(result.is_err());
        assert_eq!(calls.load(Ordering::Relaxed), 3);
        // The predicate saw "transient" on each of the first 2 retryable failures.
        let seen = seen_values.lock().unwrap().clone();
        assert_eq!(seen, vec!["transient", "transient"]);
    }

    #[tokio::test]
    async fn fn_by_name_works() {
        async fn always_ok() -> Result<&'static str, String> {
            Ok("named-fn")
        }
        let result = Retry::new().exponential(2.0).attempt(always_ok).await;
        assert_eq!(result.unwrap(), "named-fn");
    }

    #[tokio::test]
    async fn predicate_by_fn_name() {
        #[derive(Debug)]
        #[allow(dead_code)]
        enum TestError {
            Transient,
            Fatal,
        }
        impl TestError {
            fn is_transient(&self) -> bool {
                matches!(self, Self::Transient)
            }
        }

        let calls = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&calls);
        // TestError::is_transient passed as a method reference; no closure wrapper.
        let result: Result<&str, TestError> = Retry::new()
            .max_attempts(3)
            .initial_backoff(Duration::from_millis(1))
            .exponential(2.0)
            .attempt(move || {
                c.fetch_add(1, Ordering::Relaxed);
                async { Err::<&str, _>(TestError::Transient) }
            })
            .when(TestError::is_transient)
            .await;
        assert!(matches!(result.unwrap_err(), TestError::Transient));
        assert_eq!(
            calls.load(Ordering::Relaxed),
            3,
            "transient errors retried to exhaustion"
        );
    }

    #[tokio::test]
    async fn mutable_closure_state_via_move() {
        // u32 is Copy so `move` captures a copy; state++ advances on each call.
        let mut state = 0u32;
        let result: Result<u32, String> = Retry::new()
            .max_attempts(3)
            .initial_backoff(Duration::from_millis(1))
            .exponential(2.0)
            .attempt(move || {
                state += 1;
                let n = state;
                async move { if n < 3 { Err("not yet".into()) } else { Ok(n) } }
            })
            .await;
        assert_eq!(result.unwrap(), 3);
    }

    // ── max_elapsed budget ────────────────────────────────────────────────────

    #[test]
    fn max_elapsed_zero_stops_after_first_failure() {
        let calls = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&calls);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        // block_on requires a Future; RetryFuture implements IntoFuture, not Future directly.
        let result: Result<&str, String> = rt.block_on(
            Retry::new()
                .max_attempts(10)
                .initial_backoff(Duration::from_millis(1))
                .max_elapsed(Duration::ZERO)
                .exponential(2.0)
                .attempt(move || {
                    c.fetch_add(1, Ordering::Relaxed);
                    async { Err::<&str, _>("transient".into()) }
                })
                .into_future(),
        );
        assert!(result.is_err());
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn max_elapsed_fires_before_exhausting_attempts() {
        let calls = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&calls);
        tokio::time::sleep(Duration::from_millis(2)).await;
        let result: Result<&str, String> = Retry::new()
            .max_attempts(10)
            .initial_backoff(Duration::from_millis(1))
            .max_elapsed(Duration::ZERO)
            .exponential(2.0)
            .attempt(move || {
                c.fetch_add(1, Ordering::Relaxed);
                async { Err::<&str, _>("transient".into()) }
            })
            .await;
        assert!(result.is_err());
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "budget expired; should stop after 1 attempt"
        );
    }

    #[tokio::test]
    async fn max_elapsed_none_does_not_restrict() {
        let calls = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&calls);
        let result: Result<&str, String> = Retry::new()
            .max_attempts(3)
            .initial_backoff(Duration::from_millis(1))
            .exponential(2.0)
            .attempt(move || {
                c.fetch_add(1, Ordering::Relaxed);
                async { Err::<&str, _>("fail".into()) }
            })
            .await;
        assert!(result.is_err());
        assert_eq!(calls.load(Ordering::Relaxed), 3);
    }

    /// Pre-check: budget that would be overrun by exactly one sleep stops before
    /// sleeping, not after. With a 50 ms budget and a 100 ms sleep, elapsed + delay
    /// exceeds budget immediately after the first failure.
    #[tokio::test]
    async fn max_elapsed_precheck_stops_before_sleeping() {
        let calls = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&calls);
        let result: Result<&str, String> = Retry::new()
            .max_attempts(10)
            .initial_backoff(Duration::from_millis(100))
            .max_elapsed(Duration::from_millis(50))
            .exponential(2.0)
            .attempt(move || {
                c.fetch_add(1, Ordering::Relaxed);
                async { Err::<&str, _>("transient".into()) }
            })
            .await;
        assert!(result.is_err());
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "100ms sleep > 50ms budget: pre-check must stop after 1 attempt without sleeping"
        );
    }

    // ── builder invariant guards ──────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "max_attempts must be at least 1")]
    fn panics_on_zero_attempts() {
        let _ = Retry::new().max_attempts(0);
    }

    #[test]
    #[should_panic(expected = "multiplier must be >= 1.0")]
    fn panics_on_sub_one_multiplier() {
        let _ = Retry::new().exponential(0.5);
    }

    // ── far-future saturation (sleep-cap, paused clock) ──────────────────────

    /// End-to-end: max_backoff(Duration::MAX) with 2 attempts sleeps exactly
    /// SAFE_SLEEP_CAP, matching the shared one-year clamp.
    #[tokio::test(start_paused = true)]
    async fn safe_sleep_cap_clamps_duration_max_backoff() {
        let cfg = Retry::new()
            .max_attempts(2)
            .initial_backoff(Duration::MAX)
            .max_backoff(Duration::MAX)
            .exponential(2.0);
        assert_eq!(
            cfg.delay_for_attempt(0),
            SAFE_SLEEP_CAP,
            "delay_for_attempt must clamp Duration::MAX to SAFE_SLEEP_CAP"
        );

        let calls = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&calls);
        let start = tokio::time::Instant::now();
        let result: Result<&str, String> = cfg
            .attempt(move || {
                c.fetch_add(1, Ordering::Relaxed);
                async { Err::<&str, _>("fail".into()) }
            })
            .await;
        assert!(result.is_err());
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        // Paused clock: elapsed == exactly SAFE_SLEEP_CAP (tokio::time::sleep advanced it).
        assert_eq!(start.elapsed(), SAFE_SLEEP_CAP);
    }

    /// Boundary: started.elapsed() == 0 and delay == budget, so the budget pre-check
    /// fires on equality.
    #[tokio::test(start_paused = true)]
    async fn max_elapsed_boundary_zero_elapsed_equal_budget() {
        let budget = Duration::from_millis(100);
        let calls = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&calls);
        // elapsed is exactly 0 (paused clock, no time spent in op), delay == budget.
        let result: Result<&str, String> = Retry::new()
            .max_attempts(10)
            .initial_backoff(budget)
            .max_elapsed(budget)
            .exponential(1.0)
            .attempt(move || {
                c.fetch_add(1, Ordering::Relaxed);
                async { Err::<&str, _>("transient".into()) }
            })
            .await;
        assert!(result.is_err());
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "elapsed(0) + delay(budget) >= budget: must stop after 1 attempt"
        );
    }

    // ── schedule shape assertions (compute-only, no sleeps) ──────────────────

    #[test]
    fn schedule_shape_5_attempts() {
        assert_eq!(
            exponential_delay(Duration::from_secs(1), Duration::from_secs(600), 5.0, 0),
            Duration::from_secs(1)
        );
        assert_eq!(
            exponential_delay(Duration::from_secs(1), Duration::from_secs(600), 5.0, 1),
            Duration::from_secs(5)
        );
        assert_eq!(
            exponential_delay(Duration::from_secs(1), Duration::from_secs(600), 5.0, 2),
            Duration::from_secs(25)
        );
        assert_eq!(
            exponential_delay(Duration::from_secs(1), Duration::from_secs(600), 5.0, 3),
            Duration::from_secs(125)
        );
        assert_eq!(
            exponential_delay(Duration::from_secs(1), Duration::from_secs(600), 5.0, 4),
            Duration::from_secs(600)
        );
    }

    /// multiplier=1.0 keeps delay flat at initial_backoff for all attempts.
    #[test]
    fn multiplier_one_flat_schedule() {
        let base = Duration::from_millis(200);
        let cap = Duration::from_secs(60);
        assert_eq!(exponential_delay(base, cap, 1.0, 0), base);
        assert_eq!(exponential_delay(base, cap, 1.0, 5), base);
        assert_eq!(exponential_delay(base, cap, 1.0, 50), base);
    }

    // The old loop would spin 4.29B iterations for multiplier(1.0+EPSILON) and
    // initial_backoff(ZERO); the closed form returns instantly for both.
    #[test]
    fn closed_form_guard_multiplier_one_and_base_zero_return_promptly() {
        // multiplier=1.0: flat schedule (base^0 = 1, so delay = base always).
        let start = Instant::now();
        let d = exponential_delay(
            Duration::from_millis(200),
            Duration::from_secs(60),
            1.0,
            u32::MAX,
        );
        assert!(start.elapsed() < Duration::from_millis(50));
        assert_eq!(d, Duration::from_millis(200));

        // initial_backoff=ZERO: early return at base==0.
        let start = Instant::now();
        let d = exponential_delay(Duration::ZERO, Duration::from_secs(60), 2.0, u32::MAX);
        assert!(start.elapsed() < Duration::from_millis(50));
        assert_eq!(d, Duration::ZERO);
    }

    // ── Arc-shared state across multiple closure invocations ─────────────────

    #[tokio::test]
    async fn shared_state_via_arc() {
        let log = Arc::new(std::sync::Mutex::new(Vec::<&str>::new()));
        let l = Arc::clone(&log);
        let _: Result<(), String> = Retry::new()
            .max_attempts(3)
            .initial_backoff(Duration::from_millis(1))
            .exponential(2.0)
            .attempt(move || {
                l.lock().unwrap().push("call");
                async { Err::<(), _>("fail".into()) }
            })
            .await;
        assert_eq!(log.lock().unwrap().len(), 3);
    }

    // ── notify hook ──────────────────────────────────────────────────────────

    /// notify receives (attempt_index, delay) in order across 3 attempts with a paused clock.
    #[tokio::test(start_paused = true)]
    async fn notify_called_with_correct_attempt_and_delay() {
        let events = Arc::new(std::sync::Mutex::new(Vec::<(u32, Duration)>::new()));
        let ev = Arc::clone(&events);
        let _: Result<(), String> = Retry::new()
            .max_attempts(4)
            .initial_backoff(Duration::from_millis(100))
            .max_backoff(Duration::from_secs(60))
            .exponential(2.0)
            .attempt(|| async { Err::<(), _>("fail".into()) })
            .notify(move |_err, info| {
                ev.lock().unwrap().push((info.attempt_index, info.delay));
            })
            .await;
        let got = events.lock().unwrap().clone();
        // 4 attempts gives 3 sleeps: attempt 0 (100ms), 1 (200ms), 2 (400ms).
        assert_eq!(got.len(), 3);
        assert_eq!(got[0], (0, Duration::from_millis(100)));
        assert_eq!(got[1], (1, Duration::from_millis(200)));
        assert_eq!(got[2], (2, Duration::from_millis(400)));
    }

    /// notify is NOT called on first-try success.
    #[tokio::test]
    async fn notify_not_called_on_success() {
        let count = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&count);
        let result: Result<&str, String> = Retry::new()
            .exponential(2.0)
            .attempt(|| async { Ok("ok") })
            .notify(move |_e, _info| {
                c.fetch_add(1, Ordering::Relaxed);
            })
            .await;
        assert_eq!(result.unwrap(), "ok");
        assert_eq!(count.load(Ordering::Relaxed), 0);
    }

    /// notify is NOT called when `.when` rejects the error.
    #[tokio::test]
    async fn notify_not_called_when_rejected_by_when() {
        let count = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&count);
        let _: Result<&str, String> = Retry::new()
            .max_attempts(5)
            .initial_backoff(Duration::from_millis(1))
            .exponential(2.0)
            .attempt(|| async { Err::<&str, _>("fatal".into()) })
            .when(|e: &String| !e.contains("fatal"))
            .notify(move |_e, _info| {
                c.fetch_add(1, Ordering::Relaxed);
            })
            .await;
        assert_eq!(count.load(Ordering::Relaxed), 0);
    }

    /// notify works combined with `.when` when the error IS retryable.
    #[tokio::test(start_paused = true)]
    async fn notify_combined_with_when() {
        let events = Arc::new(std::sync::Mutex::new(Vec::<u32>::new()));
        let ev = Arc::clone(&events);
        let calls = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&calls);
        let _: Result<&str, String> = Retry::new()
            .max_attempts(3)
            .initial_backoff(Duration::from_millis(10))
            .exponential(2.0)
            .attempt(move || {
                c.fetch_add(1, Ordering::Relaxed);
                async { Err::<&str, _>("transient".into()) }
            })
            .when(|e: &String| e.contains("transient"))
            .notify(move |_e, info| {
                ev.lock().unwrap().push(info.attempt_index);
            })
            .await;
        let got = events.lock().unwrap().clone();
        assert_eq!(got, vec![0, 1]);
        assert_eq!(calls.load(Ordering::Relaxed), 3);
    }

    // ── attempt_with (per-attempt index) ─────────────────────────────────────

    /// attempt_with hands the op 0, 1, 2 across three tries before success.
    #[tokio::test(start_paused = true)]
    async fn attempt_with_receives_zero_based_index() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::<u32>::new()));
        let s = Arc::clone(&seen);
        let result: Result<u32, String> = Retry::new()
            .max_attempts(4)
            .initial_backoff(Duration::from_millis(1))
            .exponential(2.0)
            .attempt_with(move |attempt| {
                s.lock().unwrap().push(attempt);
                async move {
                    if attempt < 2 {
                        Err("retry".into())
                    } else {
                        Ok(attempt)
                    }
                }
            })
            .await;
        assert_eq!(result.unwrap(), 2);
        assert_eq!(*seen.lock().unwrap(), vec![0, 1, 2]);
    }

    // ── DecorrelatedJitter ───────────────────────────────────────────────────

    /// Every drawn delay stays within [base, previous_sleep*3] and never exceeds the cap.
    #[test]
    fn decorrelated_jitter_bounds_property() {
        let base = Duration::from_millis(100);
        let cap = Duration::from_secs(20);
        let policy = DecorrelatedJitter::new(base, cap);

        for _ in 0..200 {
            let mut previous = None;
            for attempt in 0..15u32 {
                let previous_for_bound = previous.unwrap_or(base);
                let upper = (previous_for_bound * 3).min(cap);
                let d = policy.next_delay(attempt, previous);
                assert!(d >= base, "attempt {attempt}: {d:?} below base {base:?}");
                assert!(d <= cap, "attempt {attempt}: {d:?} above cap {cap:?}");
                assert!(
                    d <= upper,
                    "attempt {attempt}: {d:?} above prev*3 bound {upper:?}"
                );
                previous = Some(d);
            }
        }
    }

    /// DecorrelatedJitter drives the full executor chain: it retries to exhaustion,
    /// honors max_attempts set on the future, and notifies each retryable failure.
    #[tokio::test(start_paused = true)]
    async fn decorrelated_jitter_runs_through_executor() {
        let attempts = Arc::new(std::sync::Mutex::new(Vec::<u32>::new()));
        let a = Arc::clone(&attempts);
        let result: Result<&str, String> =
            DecorrelatedJitter::new(Duration::from_millis(10), Duration::from_secs(1))
                .attempt(|| async { Err::<&str, _>("boom".into()) })
                .max_attempts(3)
                .notify(move |_e, info| a.lock().unwrap().push(info.attempt_index))
                .await;
        assert!(result.is_err());
        assert_eq!(*attempts.lock().unwrap(), vec![0, 1]);
    }

    /// base > max_backoff: every delay must stay at or below max_backoff.
    #[test]
    fn decorrelated_jitter_clamps_when_base_exceeds_cap() {
        let base = Duration::from_secs(1);
        let cap = Duration::from_millis(100);
        let policy = DecorrelatedJitter::new(base, cap);
        for attempt in 0..5u32 {
            for _ in 0..200 {
                let d = policy.next_delay(attempt, None);
                assert!(
                    d <= cap,
                    "attempt {attempt}: {d:?} exceeds max_backoff {cap:?}"
                );
            }
        }
    }

    // ── custom user-defined Policy ───────────────────────────────────────────

    /// A user Policy compiles and runs through .when + .notify + .await.
    #[tokio::test(start_paused = true)]
    async fn custom_policy_full_chain() {
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
        }

        let delays = Arc::new(std::sync::Mutex::new(Vec::<Duration>::new()));
        let d = Arc::clone(&delays);
        let result: Result<&str, String> = Fibonacci {
            unit: Duration::from_millis(10),
        }
        .attempt(|| async { Err::<&str, _>("transient".into()) })
        .max_attempts(4)
        .when(|e: &String| e.contains("transient"))
        .notify(move |_e, info| d.lock().unwrap().push(info.delay))
        .await;
        assert!(result.is_err());
        // next_delay(0)=1u, next_delay(1)=1u, next_delay(2)=2u over the three sleeps.
        assert_eq!(
            *delays.lock().unwrap(),
            vec![
                Duration::from_millis(10),
                Duration::from_millis(10),
                Duration::from_millis(20),
            ]
        );
    }

    // ── delay_hint (server Retry-After) ──────────────────────────────────────

    #[derive(Debug)]
    struct HintedError {
        retry_after: Option<Duration>,
    }
    impl std::fmt::Display for HintedError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "hinted({:?})", self.retry_after)
        }
    }

    /// The hint replaces the policy delay: paused clock, assert the exact slept time.
    #[tokio::test(start_paused = true)]
    async fn delay_hint_overrides_policy_delay() {
        let hint = Duration::from_secs(7);
        let start = tokio::time::Instant::now();
        let result: Result<&str, HintedError> = Retry::new()
            .max_attempts(2)
            // Policy would sleep 500 ms; the 7 s hint must win.
            .initial_backoff(Duration::from_millis(500))
            .exponential(2.0)
            .attempt(move || async move {
                Err::<&str, _>(HintedError {
                    retry_after: Some(hint),
                })
            })
            .delay_hint(|e: &HintedError| e.retry_after)
            .await;
        assert!(result.is_err());
        assert_eq!(
            start.elapsed(),
            hint,
            "slept the hint, not the policy delay"
        );
    }

    /// When the hint is absent the policy delay is used unchanged.
    #[tokio::test(start_paused = true)]
    async fn delay_hint_absent_falls_back_to_policy() {
        let start = tokio::time::Instant::now();
        let _: Result<&str, HintedError> = Retry::new()
            .max_attempts(2)
            .initial_backoff(Duration::from_millis(500))
            .exponential(2.0)
            .attempt(|| async { Err::<&str, _>(HintedError { retry_after: None }) })
            .delay_hint(|e: &HintedError| e.retry_after)
            .await;
        assert_eq!(start.elapsed(), Duration::from_millis(500));
    }

    /// The hint still counts toward the max_elapsed pre-check: a hint larger than the
    /// budget stops before sleeping, after exactly one attempt.
    #[tokio::test(start_paused = true)]
    async fn delay_hint_counted_by_max_elapsed_precheck() {
        let calls = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&calls);
        let result: Result<&str, HintedError> = Retry::new()
            .max_attempts(10)
            .initial_backoff(Duration::from_millis(1))
            .max_elapsed(Duration::from_secs(1))
            .exponential(2.0)
            .attempt(move || {
                c.fetch_add(1, Ordering::Relaxed);
                async {
                    Err::<&str, _>(HintedError {
                        retry_after: Some(Duration::from_secs(60)),
                    })
                }
            })
            .delay_hint(|e: &HintedError| e.retry_after)
            .await;
        assert!(result.is_err());
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "60s hint > 1s budget: pre-check must stop after one attempt"
        );
    }

    /// A custom policy with no max_delay_hint: a Duration::MAX hint is clamped only
    /// by SAFE_SLEEP_CAP (behavior unchanged from pre-0.2.1 for custom-policy callers).
    #[tokio::test(start_paused = true)]
    async fn delay_hint_custom_policy_clamped_to_safe_sleep_cap() {
        let start = tokio::time::Instant::now();
        // Fixed has no max_backoff; max_delay_hint seeds as None, so only SAFE_SLEEP_CAP applies.
        let _: Result<&str, HintedError> = Fixed(Duration::from_millis(10))
            .attempt(|| async {
                Err::<&str, _>(HintedError {
                    retry_after: Some(Duration::MAX),
                })
            })
            .max_attempts(2)
            .delay_hint(|e: &HintedError| e.retry_after)
            .await;
        assert_eq!(
            start.elapsed(),
            SAFE_SLEEP_CAP,
            "custom policy: Duration::MAX hint clamps only to the one-year cap"
        );
    }

    /// With Retry and default max_backoff, an absurd hint is clamped to max_backoff.
    #[tokio::test(start_paused = true)]
    async fn delay_hint_default_clamped_to_max_backoff() {
        let max_backoff = Duration::from_secs(2);
        let start = tokio::time::Instant::now();
        let _: Result<&str, HintedError> = Retry::new()
            .max_attempts(2)
            .max_backoff(max_backoff)
            .exponential(2.0)
            .attempt(|| async {
                Err::<&str, _>(HintedError {
                    retry_after: Some(Duration::from_secs(86_400)),
                })
            })
            .delay_hint(|e: &HintedError| e.retry_after)
            .await;
        assert_eq!(
            start.elapsed(),
            max_backoff,
            "Retry: a huge hint is clamped to max_backoff by default"
        );
    }

    /// .max_delay_hint raises the cap: a hint between max_backoff and max_delay_hint passes through,
    /// a hint above max_delay_hint is clamped to max_delay_hint.
    #[tokio::test(start_paused = true)]
    async fn max_delay_hint_raises_cap_above_max_backoff() {
        let hint_60 = Duration::from_secs(86_400); // way above both caps
        let explicit_cap = Duration::from_secs(60);

        // hint > explicit_cap: clamps to 60s
        let start = tokio::time::Instant::now();
        let _: Result<&str, HintedError> = Retry::new()
            .max_attempts(2)
            .max_backoff(Duration::from_secs(2)) // short policy backoff
            .exponential(2.0)
            .attempt(move || async move {
                Err::<&str, _>(HintedError {
                    retry_after: Some(hint_60),
                })
            })
            .delay_hint(|e: &HintedError| e.retry_after)
            .max_delay_hint(explicit_cap)
            .await;
        assert_eq!(
            start.elapsed(),
            explicit_cap,
            "hint > max_delay_hint: must clamp to max_delay_hint"
        );

        // hint < explicit_cap: passes through unchanged
        let small_hint = Duration::from_secs(30);
        let start = tokio::time::Instant::now();
        let _: Result<&str, HintedError> = Retry::new()
            .max_attempts(2)
            .max_backoff(Duration::from_secs(2))
            .exponential(2.0)
            .attempt(move || async move {
                Err::<&str, _>(HintedError {
                    retry_after: Some(small_hint),
                })
            })
            .delay_hint(|e: &HintedError| e.retry_after)
            .max_delay_hint(explicit_cap)
            .await;
        assert_eq!(
            start.elapsed(),
            small_hint,
            "hint < max_delay_hint: passes through unchanged"
        );
    }

    // ── generic Policy path ──────────────────────────────────────────────────

    /// A generic `P: Policy` caller respects the cap supplied by the policy.
    #[tokio::test]
    async fn generic_policy_respects_configured_caps() {
        async fn run_with_policy<P: Policy + Send + 'static>(policy: P) -> u32 {
            let calls = Arc::new(AtomicU32::new(0));
            let c = Arc::clone(&calls);
            let _: Result<(), String> = policy
                .attempt(move || {
                    c.fetch_add(1, Ordering::Relaxed);
                    async { Err::<(), _>("transient".into()) }
                })
                .await;
            calls.load(Ordering::Relaxed)
        }

        let count = run_with_policy(
            Retry::new()
                .max_attempts(7)
                .initial_backoff(Duration::from_millis(1))
                .exponential(2.0),
        )
        .await;
        assert_eq!(
            count, 7,
            "generic P:Policy path must respect max_attempts(7), not the default 3"
        );
    }

    // ── Fixed policy ─────────────────────────────────────────────────────────

    #[tokio::test(start_paused = true)]
    async fn fixed_policy_flat_delay() {
        let start = tokio::time::Instant::now();
        let _: Result<(), String> = Fixed(Duration::from_millis(50))
            .attempt(|| async { Err::<(), _>("fail".into()) })
            .max_attempts(3)
            .await;
        // 2 sleeps × 50ms = 100ms.
        assert_eq!(start.elapsed(), Duration::from_millis(100));
    }

    // ── closure-as-policy (Fn(u32)->Duration) ────────────────────────────────

    #[tokio::test(start_paused = true)]
    async fn closure_policy_linear_backoff() {
        let delays = Arc::new(std::sync::Mutex::new(Vec::<Duration>::new()));
        let d = Arc::clone(&delays);
        let _: Result<(), String> = (|attempt: u32| Duration::from_millis(100) * (attempt + 1))
            .attempt(|| async { Err::<(), _>("fail".into()) })
            .max_attempts(4)
            .notify(move |_, info| d.lock().unwrap().push(info.delay))
            .await;
        assert_eq!(
            *delays.lock().unwrap(),
            vec![
                Duration::from_millis(100),
                Duration::from_millis(200),
                Duration::from_millis(300),
            ]
        );
    }

    // ── non-breaking consumer check ───────────────────────────────────────────

    /// Verify the consumer call site compiles with no `use terrier::Policy` import.
    #[tokio::test]
    async fn consumer_call_compiles_without_policy_import() {
        // No `use terrier::Policy`: Retry::attempt is an inherent method.
        let result: Result<(), String> = Retry::new()
            .max_attempts(3)
            .attempt(|| async { Ok::<_, String>(()) })
            .when(|_: &String| false)
            .await;
        assert!(result.is_ok());
    }

    // ── Budget struct ────────────────────────────────────────────────────────

    #[test]
    fn budget_default_matches_retry_default() {
        let b = Budget::default();
        assert_eq!(b.max_attempts, DEFAULT_MAX_ATTEMPTS);
        assert_eq!(b.max_elapsed, None);
    }

    #[test]
    fn retry_budget_reflects_configured_values() {
        let cfg = Retry::new()
            .max_attempts(7)
            .max_elapsed(Duration::from_secs(30));
        let b = cfg.budget();
        assert_eq!(b.max_attempts, 7);
        assert_eq!(b.max_elapsed, Some(Duration::from_secs(30)));
    }

    // ── public run() ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn run_is_callable_directly() {
        let result: Result<&str, String> = Retry::new()
            .max_attempts(2)
            .initial_backoff(Duration::from_millis(1))
            .exponential(2.0)
            .attempt(|| async { Ok("via run()") })
            .run()
            .await;
        assert_eq!(result.unwrap(), "via run()");
    }
}
