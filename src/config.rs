use std::collections::HashMap;
use std::env;

use secrecy::SecretString;

#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,
    pub auth: AuthConfig,
    pub providers: HashMap<String, ProviderConfig>,
    pub routing: RoutingConfig,
    pub cache: CacheConfig,
    pub rate_limit: RateLimitConfig,
    pub otel: OtelConfig,
}

#[derive(Clone, Default)]
pub struct AuthConfig {
    pub api_keys: Vec<SecretString>,
}

impl std::fmt::Debug for AuthConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthConfig")
            .field("api_keys", &format!("[{} key(s)]", self.api_keys.len()))
            .finish()
    }
}

#[derive(Clone)]
pub struct ProviderConfig {
    pub kind: ProviderKind,
    pub api_key: Option<SecretString>,
    pub base_url: Option<String>,
    pub models: Vec<String>,
    pub priority: u32,
}

impl std::fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderConfig")
            .field("kind", &self.kind)
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .field("base_url", &self.base_url)
            .field("models", &self.models)
            .field("priority", &self.priority)
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProviderKind {
    Openai,
    Anthropic,
    Ollama,
    Gemini,
    Mistral,
    Groq,
    Together,
    Bedrock,
    /// A provider defined in the registry — instantiated via `GenericProvider`.
    Generic(String),
}

#[derive(Debug, Clone, Default)]
pub struct RoutingConfig {
    pub default_provider: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub enabled: bool,
    pub max_size: usize,
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

#[derive(Debug, Clone)]
pub struct OtelConfig {
    pub enabled: bool,
    pub endpoint: String,
    pub service_name: String,
}

impl Default for OtelConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: "http://localhost:4317".to_string(),
            service_name: "swiftllm".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    pub enabled: bool,
    pub max_requests: u64,
    pub window_seconds: u64,
    pub providers: HashMap<String, ProviderRateLimit>,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_requests: 100,
            window_seconds: 60,
            providers: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProviderRateLimit {
    pub max_requests: u64,
    pub window_seconds: u64,
}

/// Known provider names and their env var prefixes
const PROVIDER_NAMES: &[(&str, &str)] = &[
    ("openai", "OPENAI"),
    ("anthropic", "ANTHROPIC"),
    ("gemini", "GEMINI"),
    ("mistral", "MISTRAL"),
    ("ollama", "OLLAMA"),
    ("groq", "GROQ"),
    ("together", "TOGETHER"),
    ("bedrock", "BEDROCK"),
];

fn parse_provider_kind(s: &str) -> Option<ProviderKind> {
    match s.to_lowercase().as_str() {
        "openai" => Some(ProviderKind::Openai),
        "anthropic" => Some(ProviderKind::Anthropic),
        "gemini" => Some(ProviderKind::Gemini),
        "mistral" => Some(ProviderKind::Mistral),
        "ollama" => Some(ProviderKind::Ollama),
        "groq" => Some(ProviderKind::Groq),
        "together" => Some(ProviderKind::Together),
        "bedrock" => Some(ProviderKind::Bedrock),
        other => {
            // Check the provider registry for a matching schema
            if crate::providers::registry::find_schema(other).is_some() {
                Some(ProviderKind::Generic(other.to_string()))
            } else {
                None
            }
        }
    }
}

impl Config {
    /// Load configuration from environment variables (previously loaded from .env file)
    pub fn load_from_env() -> anyhow::Result<Self> {
        let port = env::var("PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8080);

        let auth = AuthConfig {
            api_keys: env::var("AUTH_API_KEYS")
                .ok()
                .map(|v| {
                    v.split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .map(SecretString::from)
                        .collect()
                })
                .unwrap_or_default(),
        };

        let routing = RoutingConfig {
            default_provider: env::var("DEFAULT_PROVIDER").ok().filter(|s| !s.is_empty()),
        };

        let cache = CacheConfig {
            enabled: env::var("CACHE_ENABLED")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(true),
            max_size: env::var("CACHE_MAX_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1000),
            ttl_seconds: env::var("CACHE_TTL_SECONDS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(300),
        };

        let mut rate_limit = RateLimitConfig {
            enabled: env::var("RATE_LIMIT_ENABLED")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(false),
            max_requests: env::var("RATE_LIMIT_MAX_REQUESTS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(100),
            window_seconds: env::var("RATE_LIMIT_WINDOW_SECONDS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(60),
            providers: HashMap::new(),
        };

        // Parse per-provider rate limits: RATE_LIMIT_<PREFIX>_MAX_REQUESTS / _WINDOW_SECONDS
        for &(_name, prefix) in PROVIDER_NAMES {
            let max_key = format!("RATE_LIMIT_{}_MAX_REQUESTS", prefix);
            let window_key = format!("RATE_LIMIT_{}_WINDOW_SECONDS", prefix);
            if let Ok(max_str) = env::var(&max_key) {
                if let Ok(max_requests) = max_str.parse::<u64>() {
                    let window_seconds = env::var(&window_key)
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(60);
                    rate_limit.providers.insert(
                        _name.to_string(),
                        ProviderRateLimit {
                            max_requests,
                            window_seconds,
                        },
                    );
                }
            }
        }

        // Parse providers from environment
        let mut providers = HashMap::new();

        for &(name, prefix) in PROVIDER_NAMES {
            // A provider is configured if it has an API key set (or for ollama, a BASE_URL)
            let api_key = env::var(format!("{}_API_KEY", prefix))
                .ok()
                .filter(|s| !s.is_empty())
                .map(SecretString::from);
            let base_url = env::var(format!("{}_BASE_URL", prefix))
                .ok()
                .filter(|s| !s.is_empty());

            // Ollama doesn't need an API key, just check for base_url or models
            let models_str = env::var(format!("{}_MODELS", prefix))
                .ok()
                .filter(|s| !s.is_empty());

            // Bedrock uses AWS credential env vars instead of API key
            let has_bedrock_config = name == "bedrock"
                && (env::var("AWS_ACCESS_KEY_ID").is_ok()
                    || env::var("BEDROCK_REGION").is_ok()
                    || models_str.is_some());

            let has_config = api_key.is_some()
                || (name == "ollama" && (base_url.is_some() || models_str.is_some()))
                || has_bedrock_config;

            if !has_config {
                continue;
            }

            let models = models_str
                .map(|v| {
                    v.split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect()
                })
                .unwrap_or_default();

            let priority = env::var(format!("{}_PRIORITY", prefix))
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(100);

            let kind = parse_provider_kind(name).unwrap();

            providers.insert(
                name.to_string(),
                ProviderConfig {
                    kind,
                    api_key,
                    base_url,
                    models,
                    priority,
                },
            );
        }

        let otel = OtelConfig {
            enabled: env::var("OTEL_ENABLED")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(false),
            endpoint: env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "http://localhost:4317".to_string()),
            service_name: env::var("OTEL_SERVICE_NAME")
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "swiftllm".to_string()),
        };

        if providers.is_empty() {
            anyhow::bail!(
                "No providers configured. Set at least one provider's API key in your .env file.\n\
                 Example: OPENAI_API_KEY=sk-..."
            );
        }

        Ok(Config {
            port,
            auth,
            providers,
            routing,
            cache,
            rate_limit,
            otel,
        })
    }

    /// Find which provider should handle a given model name
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
                    model.starts_with("gpt-")
                        || model.starts_with("o1")
                        || model.starts_with("o3")
                        || model.starts_with("o4")
                        || model.starts_with("chatgpt-")
                }
                ProviderKind::Anthropic => model.starts_with("claude-"),
                ProviderKind::Gemini => model.starts_with("gemini-"),
                ProviderKind::Mistral => {
                    model.starts_with("mistral-")
                        || model.starts_with("codestral")
                        || model.starts_with("pixtral")
                        || model.starts_with("ministral")
                        || model.starts_with("open-mistral")
                }
                ProviderKind::Ollama => {
                    model.contains(':') // ollama models typically have "model:tag" format
                }
                ProviderKind::Groq => {
                    // Groq hosts various open-source models
                    model.starts_with("llama")
                        || model.starts_with("mixtral")
                        || model.starts_with("gemma")
                }
                ProviderKind::Together => {
                    // Together AI uses org/model format
                    model.contains('/')
                }
                ProviderKind::Bedrock => {
                    // Bedrock uses provider.model-id format
                    model.starts_with("anthropic.")
                        || model.starts_with("amazon.")
                        || model.starts_with("meta.")
                        || model.starts_with("cohere.")
                        || model.starts_with("ai21.")
                        || model.starts_with("mistral.")
                }
                ProviderKind::Generic(ref schema_name) => {
                    // Check if the model is in the schema's known_models list
                    crate::providers::registry::find_schema(schema_name)
                        .map(|s| s.known_models.contains(&model))
                        .unwrap_or(false)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_otel_config_defaults() {
        let config = OtelConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.endpoint, "http://localhost:4317");
        assert_eq!(config.service_name, "swiftllm");
    }

    /// Tests that exercise `load_from_env` with OTEL env vars are combined into
    /// a single test to avoid race conditions from parallel env var mutation.
    #[test]
    fn test_otel_config_from_env() {
        // --- Subcase 1: disabled by default ---
        env::remove_var("OTEL_ENABLED");
        env::remove_var("OTEL_EXPORTER_OTLP_ENDPOINT");
        env::remove_var("OTEL_SERVICE_NAME");
        env::set_var("OPENAI_API_KEY", "sk-test-key");

        let config = Config::load_from_env().unwrap();
        assert!(!config.otel.enabled);
        assert_eq!(config.otel.endpoint, "http://localhost:4317");
        assert_eq!(config.otel.service_name, "swiftllm");

        // --- Subcase 2: enabled with custom values ---
        env::set_var("OTEL_ENABLED", "true");
        env::set_var("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317");
        env::set_var("OTEL_SERVICE_NAME", "my-gateway");

        let config = Config::load_from_env().unwrap();
        assert!(config.otel.enabled);
        assert_eq!(config.otel.endpoint, "http://collector:4317");
        assert_eq!(config.otel.service_name, "my-gateway");

        // --- Subcase 3: empty strings fall back to defaults ---
        env::set_var("OTEL_ENABLED", "false");
        env::set_var("OTEL_EXPORTER_OTLP_ENDPOINT", "");
        env::set_var("OTEL_SERVICE_NAME", "");

        let config = Config::load_from_env().unwrap();
        assert!(!config.otel.enabled);
        assert_eq!(config.otel.endpoint, "http://localhost:4317");
        assert_eq!(config.otel.service_name, "swiftllm");

        // Cleanup
        env::remove_var("OTEL_ENABLED");
        env::remove_var("OTEL_EXPORTER_OTLP_ENDPOINT");
        env::remove_var("OTEL_SERVICE_NAME");
        env::remove_var("OPENAI_API_KEY");
    }
}
