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

    // TODO: add cache config
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct AuthConfig {
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
    pub default_provider: Option<String>,
}

fn default_port() -> u16 { 8080 }
fn default_base_url_none() -> Option<String> { None }
fn default_priority() -> u32 { 100 }

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read config file {}: {}", path.display(), e))?;
        let config: Config = toml::from_str(&content)
            .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;
        Ok(config)
    }

    pub fn find_provider_for_model(&self, model: &str) -> Option<(&String, &ProviderConfig)> {
        // First: exact match in provider model lists
        for (name, provider) in &self.providers {
            if provider.models.iter().any(|m| m == model) {
                return Some((name, provider));
            }
        }

        // Second: prefix-based matching
        for (name, provider) in &self.providers {
            let matches = match provider.kind {
                ProviderKind::Openai => {
                    model.starts_with("gpt-") || model.starts_with("o1") || model.starts_with("o3") || model.starts_with("o4")
                }
                ProviderKind::Anthropic => {
                    model.starts_with("claude-")
                }
                ProviderKind::Ollama => {
                    model.contains(':')
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

    pub fn providers_by_priority(&self) -> Vec<(&String, &ProviderConfig)> {
        let mut providers: Vec<_> = self.providers.iter().collect();
        providers.sort_by_key(|(_, p)| p.priority);
        providers
    }
}
