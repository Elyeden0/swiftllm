use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    #[serde(default = "default_port")]
    pub port: u16,

    #[serde(default)]
    pub auth: AuthConfig,

    pub providers: HashMap<String, ProviderConfig>,

    #[serde(default)]
    pub routing: RoutingConfig,

    #[serde(default)]
    pub cache: CacheConfig,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct AuthConfig {
    /// API keys that clients must provide to use the proxy
    #[serde(default)]
    pub api_keys: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ProviderConfig {
    pub kind: ProviderKind,
    pub api_key: Option<String>,
    #[serde(default = "default_base_url_none")]
    pub base_url: Option<String>,
    #[serde(default)]
    pub models: Vec<String>,
    /// Priority for failover (lower = higher priority)
    #[serde(default = "default_priority")]
    pub priority: u32,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    Openai,
    Anthropic,
    Ollama,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct RoutingConfig {
    /// Fallback provider name if model not found in any provider's model list
    pub default_provider: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CacheConfig {
    #[serde(default = "default_cache_enabled")]
    pub enabled: bool,
    #[serde(default = "default_cache_max_size")]
    pub max_size: usize,
    #[serde(default = "default_cache_ttl")]
    pub ttl_seconds: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_size: 1000,
            ttl_seconds: 300,
        }
    }
}

fn default_cache_enabled() -> bool {
    true
}

fn default_cache_max_size() -> usize {
    1000
}

fn default_cache_ttl() -> u64 {
    300
}

fn default_port() -> u16 {
    8080
}

fn default_base_url_none() -> Option<String> {
    None
}

fn default_priority() -> u32 {
    100
}

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read config file {}: {}", path.display(), e))?;
        let config: Config = toml::from_str(&content)
            .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
        Ok(config)
    }

    /// Find which provider should handle a given model name
    pub fn find_provider_for_model(&self, model: &str) -> Option<(&String, &ProviderConfig)> {
        // First: exact match in provider model lists
        for (name, provider) in &self.providers {
            if provider.models.iter().any(|m| m == model) {
                return Some((name, provider));
            }
        }

        // Second: prefix-based matching (e.g., "claude-" → anthropic, "gpt-" → openai)
        for (name, provider) in &self.providers {
            let matches = match provider.kind {
                ProviderKind::Openai => {
                    model.starts_with("gpt-") || model.starts_with("o1") || model.starts_with("o3") || model.starts_with("o4")
                }
                ProviderKind::Anthropic => {
                    model.starts_with("claude-")
                }
                ProviderKind::Ollama => {
                    model.contains(':') // ollama models typically have "model:tag" format
                }
            };
            if matches {
                return Some((name, provider));
            }
        }

        // Third: default provider
        if let Some(ref default_name) = self.routing.default_provider {
            if let Some(provider) = self.providers.get(default_name) {
                return Some((default_name, provider));
            }
        }

        None
    }

    /// Get providers sorted by priority for failover
    pub fn providers_by_priority(&self) -> Vec<(&String, &ProviderConfig)> {
        let mut providers: Vec<_> = self.providers.iter().collect();
        providers.sort_by_key(|(_, p)| p.priority);
        providers
    }
}
