pub mod latency;
pub mod quality;
pub mod router;

use serde::{Deserialize, Serialize};

// ── Smart routing request types ─────────────────────────────────────────────

/// Configuration for smart model routing on a per-request basis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartRoutingConfig {
    pub strategy: RoutingStrategy,
    #[serde(default)]
    pub quality: QualityTier,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_cost_per_1k_tokens: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RoutingStrategy {
    CostOptimized,
    LatencyOptimized,
    Balanced,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default, Hash)]
#[serde(rename_all = "snake_case")]
pub enum QualityTier {
    Low,
    #[default]
    Medium,
    High,
}

// ── Routing metadata in response ────────────────────────────────────────────

/// Metadata attached to responses when smart routing was used.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingMetadata {
    pub strategy: String,
    pub selected_model: String,
    pub selected_provider: String,
    pub reason: String,
    pub alternatives_considered: usize,
}
