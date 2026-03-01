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

    // TODO: add routing config (default provider)
    // TODO: add cache config
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
    // TODO: add priority for failover
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    Openai,
    Anthropic,
    Ollama,
}

fn default_port() -> u16 { 8080 }
fn default_base_url_none() -> Option<String> { None }

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read config file {}: {}", path.display(), e))?;
        let config: Config = toml::from_str(&content)
            .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
        Ok(config)
    }

    pub fn find_provider_for_model(&self, model: &str) -> Option<(&String, &ProviderConfig)> {
        // Exact match only for now
        // TODO: add prefix-based matching
        for (name, provider) in &self.providers {
            if provider.models.iter().any(|m| m == model) {
                return Some((name, provider));
            }
        }
        None
    }
}
