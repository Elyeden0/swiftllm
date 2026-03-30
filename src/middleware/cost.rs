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

impl Default for CostTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl CostTracker {
    pub fn new() -> Self {
        let mut pricing = HashMap::new();

        // ── OpenAI pricing (approximate, March 2026) ────────────────────
        pricing.insert(
            "gpt-4o".into(),
            ModelPricing {
                input_per_million: 2.50,
                output_per_million: 10.00,
            },
        );
        pricing.insert(
            "gpt-4o-mini".into(),
            ModelPricing {
                input_per_million: 0.15,
                output_per_million: 0.60,
            },
        );
        pricing.insert(
            "gpt-4-turbo".into(),
            ModelPricing {
                input_per_million: 10.00,
                output_per_million: 30.00,
            },
        );
        pricing.insert(
            "o3-mini".into(),
            ModelPricing {
                input_per_million: 1.10,
                output_per_million: 4.40,
            },
        );
        pricing.insert(
            "gpt-4.1".into(),
            ModelPricing {
                input_per_million: 2.00,
                output_per_million: 8.00,
            },
        );
        pricing.insert(
            "gpt-4.1-mini".into(),
            ModelPricing {
                input_per_million: 0.40,
                output_per_million: 1.60,
            },
        );
        pricing.insert(
            "gpt-4.1-nano".into(),
            ModelPricing {
                input_per_million: 0.10,
                output_per_million: 0.40,
            },
        );
        pricing.insert(
            "o3".into(),
            ModelPricing {
                input_per_million: 2.00,
                output_per_million: 8.00,
            },
        );
        pricing.insert(
            "o4-mini".into(),
            ModelPricing {
                input_per_million: 1.10,
                output_per_million: 4.40,
            },
        );

        // ── OpenAI embedding pricing ────────────────────────────────────
        pricing.insert(
            "text-embedding-3-small".into(),
            ModelPricing {
                input_per_million: 0.02,
                output_per_million: 0.0,
            },
        );
        pricing.insert(
            "text-embedding-3-large".into(),
            ModelPricing {
                input_per_million: 0.13,
                output_per_million: 0.0,
            },
        );
        pricing.insert(
            "text-embedding-ada-002".into(),
            ModelPricing {
                input_per_million: 0.10,
                output_per_million: 0.0,
            },
        );

        // ── Mistral embedding pricing ──────────────────────────────────
        pricing.insert(
            "mistral-embed".into(),
            ModelPricing {
                input_per_million: 0.10,
                output_per_million: 0.0,
            },
        );

        // ── Anthropic pricing ───────────────────────────────────────────
        pricing.insert(
            "claude-opus-4-6".into(),
            ModelPricing {
                input_per_million: 15.00,
                output_per_million: 75.00,
            },
        );
        pricing.insert(
            "claude-sonnet-4-6".into(),
            ModelPricing {
                input_per_million: 3.00,
                output_per_million: 15.00,
            },
        );
        pricing.insert(
            "claude-haiku-4-5-20251001".into(),
            ModelPricing {
                input_per_million: 0.80,
                output_per_million: 4.00,
            },
        );

        // ── Google Gemini pricing ───────────────────────────────────────
        pricing.insert(
            "gemini-2.0-flash".into(),
            ModelPricing {
                input_per_million: 0.10,
                output_per_million: 0.40,
            },
        );
        pricing.insert(
            "gemini-2.0-pro".into(),
            ModelPricing {
                input_per_million: 1.25,
                output_per_million: 10.00,
            },
        );
        pricing.insert(
            "gemini-1.5-pro".into(),
            ModelPricing {
                input_per_million: 1.25,
                output_per_million: 5.00,
            },
        );
        pricing.insert(
            "gemini-1.5-flash".into(),
            ModelPricing {
                input_per_million: 0.075,
                output_per_million: 0.30,
            },
        );

        // ── Mistral pricing ────────────────────────────────────────────
        pricing.insert(
            "mistral-large-latest".into(),
            ModelPricing {
                input_per_million: 2.00,
                output_per_million: 6.00,
            },
        );
        pricing.insert(
            "mistral-medium-latest".into(),
            ModelPricing {
                input_per_million: 2.70,
                output_per_million: 8.10,
            },
        );
        pricing.insert(
            "mistral-small-latest".into(),
            ModelPricing {
                input_per_million: 0.20,
                output_per_million: 0.60,
            },
        );
        pricing.insert(
            "codestral-latest".into(),
            ModelPricing {
                input_per_million: 0.30,
                output_per_million: 0.90,
            },
        );
        pricing.insert(
            "ministral-8b-latest".into(),
            ModelPricing {
                input_per_million: 0.10,
                output_per_million: 0.10,
            },
        );
        pricing.insert(
            "pixtral-large-latest".into(),
            ModelPricing {
                input_per_million: 2.00,
                output_per_million: 6.00,
            },
        );

        // ── Groq pricing (LPU inference) ───────────────────────────────
        pricing.insert(
            "llama-3.3-70b-versatile".into(),
            ModelPricing {
                input_per_million: 0.59,
                output_per_million: 0.79,
            },
        );
        pricing.insert(
            "llama-3.1-8b-instant".into(),
            ModelPricing {
                input_per_million: 0.05,
                output_per_million: 0.08,
            },
        );
        pricing.insert(
            "mixtral-8x7b-32768".into(),
            ModelPricing {
                input_per_million: 0.24,
                output_per_million: 0.24,
            },
        );
        pricing.insert(
            "gemma2-9b-it".into(),
            ModelPricing {
                input_per_million: 0.20,
                output_per_million: 0.20,
            },
        );

        // ── Together AI pricing ─────────────────────────────────────────
        pricing.insert(
            "meta-llama/Llama-3.3-70B-Instruct-Turbo".into(),
            ModelPricing {
                input_per_million: 0.88,
                output_per_million: 0.88,
            },
        );
        pricing.insert(
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo".into(),
            ModelPricing {
                input_per_million: 0.18,
                output_per_million: 0.18,
            },
        );
        pricing.insert(
            "mistralai/Mixtral-8x7B-Instruct-v0.1".into(),
            ModelPricing {
                input_per_million: 0.60,
                output_per_million: 0.60,
            },
        );
        pricing.insert(
            "Qwen/Qwen2.5-72B-Instruct-Turbo".into(),
            ModelPricing {
                input_per_million: 1.20,
                output_per_million: 1.20,
            },
        );

        // ── AWS Bedrock pricing (Claude models) ─────────────────────────
        pricing.insert(
            "anthropic.claude-3-5-sonnet-20241022-v2:0".into(),
            ModelPricing {
                input_per_million: 3.00,
                output_per_million: 15.00,
            },
        );
        pricing.insert(
            "anthropic.claude-3-haiku-20240307-v1:0".into(),
            ModelPricing {
                input_per_million: 0.25,
                output_per_million: 1.25,
            },
        );
        pricing.insert(
            "amazon.titan-text-premier-v1:0".into(),
            ModelPricing {
                input_per_million: 0.50,
                output_per_million: 1.50,
            },
        );
        pricing.insert(
            "meta.llama3-1-70b-instruct-v1:0".into(),
            ModelPricing {
                input_per_million: 0.99,
                output_per_million: 0.99,
            },
        );

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
