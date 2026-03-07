use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Token bucket rate limiter per provider
pub struct RateLimiter {
    buckets: Mutex<HashMap<String, TokenBucket>>,
    limits: HashMap<String, RateLimitConfig>,
    /// Global fallback limit if no per-provider config is set
    default_limit: Option<RateLimitConfig>,
}

#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum requests allowed in the window
    pub max_requests: u64,
    /// Time window duration
    pub window: Duration,
}

struct TokenBucket {
    tokens: f64,
    max_tokens: f64,
    refill_rate: f64, // tokens per second
    last_refill: Instant,
}

impl TokenBucket {
    fn new(config: &RateLimitConfig) -> Self {
        let max_tokens = config.max_requests as f64;
        let refill_rate = max_tokens / config.window.as_secs_f64();
        Self {
            tokens: max_tokens,
            max_tokens,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    /// Try to consume one token. Returns true if allowed.
    fn try_acquire(&mut self) -> bool {
        self.refill();
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Refill tokens based on elapsed time
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.max_tokens);
        self.last_refill = now;
    }

    /// Seconds until the next token is available
    fn retry_after(&self) -> f64 {
        if self.tokens >= 1.0 {
            return 0.0;
        }
        let needed = 1.0 - self.tokens;
        needed / self.refill_rate
    }
}

impl RateLimiter {
    pub fn new(
        limits: HashMap<String, RateLimitConfig>,
        default_limit: Option<RateLimitConfig>,
    ) -> Self {
        Self {
            buckets: Mutex::new(HashMap::new()),
            limits,
            default_limit,
        }
    }

    /// Check if a request to the given provider is allowed.
    /// Returns Ok(()) if allowed, Err(retry_after_seconds) if rate limited.
    pub fn check(&self, provider: &str) -> Result<(), f64> {
        // Find the config for this provider, or use default
        let config = self.limits.get(provider).or(self.default_limit.as_ref());

        let config = match config {
            Some(c) => c,
            None => return Ok(()), // no limit configured → allow
        };

        let mut buckets = self.buckets.lock().unwrap();
        let bucket = buckets
            .entry(provider.to_string())
            .or_insert_with(|| TokenBucket::new(config));

        if bucket.try_acquire() {
            Ok(())
        } else {
            Err(bucket.retry_after())
        }
    }

    /// Get current rate limit stats for all providers
    pub fn stats(&self) -> HashMap<String, RateLimitStats> {
        let mut buckets = self.buckets.lock().unwrap();
        buckets
            .iter_mut()
            .map(|(name, bucket)| {
                bucket.refill();
                (
                    name.clone(),
                    RateLimitStats {
                        remaining: bucket.tokens as u64,
                        limit: bucket.max_tokens as u64,
                    },
                )
            })
            .collect()
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RateLimitStats {
    pub remaining: u64,
    pub limit: u64,
}
