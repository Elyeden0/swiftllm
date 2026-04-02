use std::collections::HashMap;

use super::QualityTier;

/// Static lookup table mapping known model names to quality tiers.
pub fn build_quality_map() -> HashMap<&'static str, QualityTier> {
    let mut m = HashMap::new();

    // ── Low tier ────────────────────────────────────────────────────────────
    m.insert("claude-haiku-4-5-20251001", QualityTier::Low);
    m.insert("gpt-4o-mini", QualityTier::Low);
    m.insert("gpt-4.1-mini", QualityTier::Low);
    m.insert("gpt-4.1-nano", QualityTier::Low);
    m.insert("gemini-2.0-flash", QualityTier::Low);
    m.insert("gemini-1.5-flash", QualityTier::Low);
    m.insert("mistral-small-latest", QualityTier::Low);
    m.insert("ministral-8b-latest", QualityTier::Low);
    m.insert("codestral-latest", QualityTier::Low);
    m.insert("llama-3.1-8b-instant", QualityTier::Low);
    m.insert("gemma2-9b-it", QualityTier::Low);
    m.insert("o3-mini", QualityTier::Low);
    m.insert("o4-mini", QualityTier::Low);
    m.insert(
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        QualityTier::Low,
    );
    m.insert("anthropic.claude-3-haiku-20240307-v1:0", QualityTier::Low);

    // ── Medium tier ─────────────────────────────────────────────────────────
    m.insert("claude-sonnet-4-6", QualityTier::Medium);
    m.insert("gpt-4o", QualityTier::Medium);
    m.insert("gpt-4.1", QualityTier::Medium);
    m.insert("gpt-4-turbo", QualityTier::Medium);
    m.insert("gemini-2.0-pro", QualityTier::Medium);
    m.insert("gemini-1.5-pro", QualityTier::Medium);
    m.insert("mistral-large-latest", QualityTier::Medium);
    m.insert("mistral-medium-latest", QualityTier::Medium);
    m.insert("pixtral-large-latest", QualityTier::Medium);
    m.insert("llama-3.3-70b-versatile", QualityTier::Medium);
    m.insert("mixtral-8x7b-32768", QualityTier::Medium);
    m.insert(
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        QualityTier::Medium,
    );
    m.insert("mistralai/Mixtral-8x7B-Instruct-v0.1", QualityTier::Medium);
    m.insert("Qwen/Qwen2.5-72B-Instruct-Turbo", QualityTier::Medium);
    m.insert(
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        QualityTier::Medium,
    );
    m.insert("amazon.titan-text-premier-v1:0", QualityTier::Medium);
    m.insert("meta.llama3-1-70b-instruct-v1:0", QualityTier::Medium);

    // ── High tier ───────────────────────────────────────────────────────────
    m.insert("claude-opus-4-6", QualityTier::High);
    m.insert("o3", QualityTier::High);

    m
}

/// Look up the quality tier for a model. Defaults to Medium for unknown models.
pub fn tier_for_model(model: &str) -> QualityTier {
    // We rebuild each time for simplicity; in a hot path you'd cache this.
    let map = build_quality_map();
    map.get(model).cloned().unwrap_or(QualityTier::Medium)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_low_tier() {
        assert_eq!(tier_for_model("gpt-4o-mini"), QualityTier::Low);
    }

    #[test]
    fn known_medium_tier() {
        assert_eq!(tier_for_model("claude-sonnet-4-6"), QualityTier::Medium);
    }

    #[test]
    fn known_high_tier() {
        assert_eq!(tier_for_model("claude-opus-4-6"), QualityTier::High);
    }

    #[test]
    fn unknown_defaults_to_medium() {
        assert_eq!(tier_for_model("some-unknown-model"), QualityTier::Medium);
    }
}
