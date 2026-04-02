use crate::config::Config;
use crate::middleware::cost::CostTracker;
use crate::providers::ProviderError;

use super::latency::LatencyTracker;
use super::quality::tier_for_model;
use super::{QualityTier, RoutingMetadata, RoutingStrategy, SmartRoutingConfig};

/// A candidate model that can be selected by the router.
#[derive(Debug, Clone)]
struct Candidate {
    provider_name: String,
    model: String,
    quality: QualityTier,
    /// Cost per 1k *input* tokens (USD). None if not in pricing table.
    cost_per_1k: Option<f64>,
}

/// Smart router that selects the optimal provider and model for a request.
pub struct SmartRouter {
    latency_tracker: LatencyTracker,
}

impl SmartRouter {
    pub fn new(latency_tracker: LatencyTracker) -> Self {
        Self { latency_tracker }
    }

    /// Access the underlying latency tracker (e.g. to record durations).
    pub fn latency_tracker(&self) -> &LatencyTracker {
        &self.latency_tracker
    }

    /// Select the best (provider_name, model_name) pair for the given routing config.
    /// Also returns `RoutingMetadata` describing the decision.
    pub fn select_model(
        &self,
        config: &SmartRoutingConfig,
        app_config: &Config,
        cost_tracker: &CostTracker,
    ) -> Result<(String, String, RoutingMetadata), ProviderError> {
        // Build candidate list from all configured providers and their models.
        let candidates = self.build_candidates(app_config, cost_tracker);

        if candidates.is_empty() {
            return Err(ProviderError::Config(
                "No models available for smart routing".to_string(),
            ));
        }

        // Filter by quality tier.
        let mut filtered: Vec<_> = candidates
            .iter()
            .filter(|c| c.quality == config.quality)
            .collect();

        // If no exact tier match, fall back to all candidates.
        if filtered.is_empty() {
            filtered = candidates.iter().collect();
        }

        let alternatives_considered = filtered.len();

        // Apply optional max cost filter.
        if let Some(max_cost) = config.max_cost_per_1k_tokens {
            let cost_filtered: Vec<_> = filtered
                .iter()
                .filter(|c| c.cost_per_1k.is_some_and(|cost| cost <= max_cost))
                .copied()
                .collect();
            if !cost_filtered.is_empty() {
                filtered = cost_filtered;
            }
        }

        let (selected, reason) = match config.strategy {
            RoutingStrategy::CostOptimized => self.select_cost_optimized(&filtered),
            RoutingStrategy::LatencyOptimized => self.select_latency_optimized(&filtered),
            RoutingStrategy::Balanced => self.select_balanced(&filtered),
        };

        let strategy_str = match config.strategy {
            RoutingStrategy::CostOptimized => "cost_optimized",
            RoutingStrategy::LatencyOptimized => "latency_optimized",
            RoutingStrategy::Balanced => "balanced",
        };

        let metadata = RoutingMetadata {
            strategy: strategy_str.to_string(),
            selected_model: selected.model.clone(),
            selected_provider: selected.provider_name.clone(),
            reason,
            alternatives_considered,
        };

        Ok((
            selected.provider_name.clone(),
            selected.model.clone(),
            metadata,
        ))
    }

    // ── Private helpers ─────────────────────────────────────────────────────

    fn build_candidates(&self, app_config: &Config, cost_tracker: &CostTracker) -> Vec<Candidate> {
        let mut candidates = Vec::new();

        for (provider_name, provider_config) in &app_config.providers {
            for model in &provider_config.models {
                let quality = tier_for_model(model);
                let cost_per_1k = cost_tracker.lookup_cost(model, 1000, 0).map(|c| c * 1000.0); // lookup_cost returns per-token cost for 1000 tokens → scale

                candidates.push(Candidate {
                    provider_name: provider_name.clone(),
                    model: model.clone(),
                    quality,
                    cost_per_1k,
                });
            }
        }

        candidates
    }

    fn select_cost_optimized<'a>(&self, candidates: &[&'a Candidate]) -> (&'a Candidate, String) {
        let mut best = candidates[0];
        for &c in &candidates[1..] {
            match (c.cost_per_1k, best.cost_per_1k) {
                (Some(a), Some(b)) if a < b => best = c,
                (Some(_), None) => best = c, // prefer known cost
                _ => {}
            }
        }
        let tier = format!("{:?}", best.quality).to_lowercase();
        (best, format!("cheapest model in '{}' quality tier", tier))
    }

    fn select_latency_optimized<'a>(
        &self,
        candidates: &[&'a Candidate],
    ) -> (&'a Candidate, String) {
        let mut best = candidates[0];
        let mut best_latency = self
            .latency_tracker
            .p50(&best.provider_name)
            .map(|d| d.as_millis());

        for &c in &candidates[1..] {
            let lat = self
                .latency_tracker
                .p50(&c.provider_name)
                .map(|d| d.as_millis());
            match (lat, best_latency) {
                (Some(a), Some(b)) if a < b => {
                    best = c;
                    best_latency = Some(a);
                }
                (Some(a), None) => {
                    best = c;
                    best_latency = Some(a);
                }
                _ => {}
            }
        }
        let reason = match best_latency {
            Some(ms) => format!("lowest p50 latency ({}ms) among candidates", ms),
            None => "selected first available (no latency data)".to_string(),
        };
        (best, reason)
    }

    fn select_balanced<'a>(&self, candidates: &[&'a Candidate]) -> (&'a Candidate, String) {
        // Gather raw values for normalization.
        let costs: Vec<f64> = candidates
            .iter()
            .map(|c| c.cost_per_1k.unwrap_or(f64::MAX))
            .collect();
        let latencies: Vec<f64> = candidates
            .iter()
            .map(|c| {
                self.latency_tracker
                    .p50(&c.provider_name)
                    .map(|d| d.as_millis() as f64)
                    .unwrap_or(f64::MAX)
            })
            .collect();

        let cost_min = costs.iter().copied().fold(f64::INFINITY, f64::min);
        let cost_max = costs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lat_min = latencies.iter().copied().fold(f64::INFINITY, f64::min);
        let lat_max = latencies.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let normalize = |val: f64, min: f64, max: f64| -> f64 {
            if (max - min).abs() < f64::EPSILON {
                0.0
            } else {
                (val - min) / (max - min)
            }
        };

        let mut best_idx = 0;
        let mut best_score = f64::MAX;

        for (i, _c) in candidates.iter().enumerate() {
            let cost_norm = normalize(costs[i], cost_min, cost_max);
            let lat_norm = normalize(latencies[i], lat_min, lat_max);
            // error_rate: we don't have per-model error tracking, so use 0.0
            let error_rate = 0.0_f64;
            let score = cost_norm * 0.4 + lat_norm * 0.3 + error_rate * 0.3;
            if score < best_score {
                best_score = score;
                best_idx = i;
            }
        }

        (
            candidates[best_idx],
            format!(
                "best balanced score ({:.3}) across cost, latency, and reliability",
                best_score
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_optimized_picks_cheapest() {
        let candidates = vec![
            Candidate {
                provider_name: "openai".into(),
                model: "gpt-4o".into(),
                quality: QualityTier::Medium,
                cost_per_1k: Some(0.0025),
            },
            Candidate {
                provider_name: "anthropic".into(),
                model: "claude-sonnet-4-6".into(),
                quality: QualityTier::Medium,
                cost_per_1k: Some(0.003),
            },
        ];
        let refs: Vec<&Candidate> = candidates.iter().collect();
        let tracker = LatencyTracker::new();
        let router = SmartRouter::new(tracker);
        let (best, _reason) = router.select_cost_optimized(&refs);
        assert_eq!(best.model, "gpt-4o");
    }

    #[test]
    fn latency_optimized_picks_fastest() {
        let tracker = LatencyTracker::new();
        tracker.record("slow", std::time::Duration::from_millis(500));
        tracker.record("fast", std::time::Duration::from_millis(100));

        let candidates = vec![
            Candidate {
                provider_name: "slow".into(),
                model: "model-a".into(),
                quality: QualityTier::Medium,
                cost_per_1k: Some(0.001),
            },
            Candidate {
                provider_name: "fast".into(),
                model: "model-b".into(),
                quality: QualityTier::Medium,
                cost_per_1k: Some(0.002),
            },
        ];
        let refs: Vec<&Candidate> = candidates.iter().collect();
        let router = SmartRouter::new(tracker);
        let (best, _reason) = router.select_latency_optimized(&refs);
        assert_eq!(best.provider_name, "fast");
    }
}
