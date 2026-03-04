use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

/// Per-model pricing (USD per 1M tokens)
#[derive(Debug, Clone)]
struct ModelPricing {
    input_per_million: f64,
    output_per_million: f64,
}

/// Tracks token usage and costs across all requests
pub struct CostTracker {
    inner: Mutex<CostTrackerInner>,
}

struct CostTrackerInner {
    /// Per-provider, per-model stats
    stats: HashMap<String, ProviderStats>,
    /// Known model pricing
    pricing: HashMap<String, ModelPricing>,
    /// Total request count
    total_requests: u64,
    /// Total cache hits
    cache_hits: u64,
    /// Total errors
    total_errors: u64,
    /// Start time
    started_at: u64,
}

#[derive(Debug, Clone, Default)]
struct ProviderStats {
    requests: u64,
    input_tokens: u64,
    output_tokens: u64,
    total_cost_usd: f64,
    errors: u64,
}

impl CostTracker {
    pub fn new() -> Self {
        let mut pricing = HashMap::new();

        // OpenAI pricing (approximate, March 2026)
        pricing.insert("gpt-4o".into(), ModelPricing { input_per_million: 2.50, output_per_million: 10.00 });
        pricing.insert("gpt-4o-mini".into(), ModelPricing { input_per_million: 0.15, output_per_million: 0.60 });
        pricing.insert("gpt-4-turbo".into(), ModelPricing { input_per_million: 10.00, output_per_million: 30.00 });
        pricing.insert("o3-mini".into(), ModelPricing { input_per_million: 1.10, output_per_million: 4.40 });

        // Anthropic pricing
        pricing.insert("claude-opus-4-6".into(), ModelPricing { input_per_million: 15.00, output_per_million: 75.00 });
        pricing.insert("claude-sonnet-4-6".into(), ModelPricing { input_per_million: 3.00, output_per_million: 15.00 });
        pricing.insert("claude-haiku-4-5-20251001".into(), ModelPricing { input_per_million: 0.80, output_per_million: 4.00 });

        Self {
            inner: Mutex::new(CostTrackerInner {
                stats: HashMap::new(),
                pricing,
                total_requests: 0,
                cache_hits: 0,
                total_errors: 0,
                started_at: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            }),
        }
    }

    /// Record a completed request
    pub fn record_request(
        &self,
        provider: &str,
        model: &str,
        input_tokens: u64,
        output_tokens: u64,
    ) {
        let mut inner = self.inner.lock().unwrap();
        inner.total_requests += 1;

        // Calculate cost first (immutable borrow of pricing)
        let cost = inner.pricing.get(model).map(|pricing| {
            (input_tokens as f64 * pricing.input_per_million
                + output_tokens as f64 * pricing.output_per_million)
                / 1_000_000.0
        });

        // Then update stats (mutable borrow)
        let stats = inner.stats.entry(provider.to_string()).or_default();
        stats.requests += 1;
        stats.input_tokens += input_tokens;
        stats.output_tokens += output_tokens;
        if let Some(cost) = cost {
            stats.total_cost_usd += cost;
        }
    }

    /// Record a cache hit
    pub fn record_cache_hit(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.cache_hits += 1;
        inner.total_requests += 1;
    }

    /// Record an error
    pub fn record_error(&self, provider: &str) {
        let mut inner = self.inner.lock().unwrap();
        inner.total_errors += 1;
        let stats = inner.stats.entry(provider.to_string()).or_default();
        stats.errors += 1;
    }

    /// Get a snapshot of all stats
    pub fn snapshot(&self) -> StatsSnapshot {
        let inner = self.inner.lock().unwrap();

        let providers: HashMap<String, ProviderStatsSnapshot> = inner
            .stats
            .iter()
            .map(|(name, stats)| {
                (
                    name.clone(),
                    ProviderStatsSnapshot {
                        requests: stats.requests,
                        input_tokens: stats.input_tokens,
                        output_tokens: stats.output_tokens,
                        total_cost_usd: (stats.total_cost_usd * 10000.0).round() / 10000.0,
                        errors: stats.errors,
                    },
                )
            })
            .collect();

        let total_cost: f64 = inner.stats.values().map(|s| s.total_cost_usd).sum();

        StatsSnapshot {
            total_requests: inner.total_requests,
            cache_hits: inner.cache_hits,
            total_errors: inner.total_errors,
            total_cost_usd: (total_cost * 10000.0).round() / 10000.0,
            uptime_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                - inner.started_at,
            providers,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StatsSnapshot {
    pub total_requests: u64,
    pub cache_hits: u64,
    pub total_errors: u64,
    pub total_cost_usd: f64,
    pub uptime_seconds: u64,
    pub providers: HashMap<String, ProviderStatsSnapshot>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProviderStatsSnapshot {
    pub requests: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_cost_usd: f64,
    pub errors: u64,
}
