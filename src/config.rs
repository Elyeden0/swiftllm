use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    #[serde(default = "default_port")]
    pub port: u16,

    pub providers: HashMap<String, ProviderConfig>,

    // TODO: add auth config (api keys for proxy access)
    // TODO: add routing config (default provider, fallback)
    // TODO: add cache config
}

#[derive(Debug, Deserialize, Clone)]
pub struct ProviderConfig {
    pub kind: ProviderKind,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    #[serde(default)]
    pub models: Vec<String>,
    // TODO: add priority for failover ordering
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    Openai,
    Anthropic,
    Ollama,
}

fn default_port() -> u16 {
    8080
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
        // Exact match only for now
        // TODO: add prefix-based matching (gpt-* -> openai, claude-* -> anthropic)
        for (name, provider) in &self.providers {
            if provider.models.iter().any(|m| m == model) {
                return Some((name, provider));
            }
        }
        None
    }
}
