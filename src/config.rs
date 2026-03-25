use std::collections::HashMap;
use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,
    pub auth: AuthConfig,
    pub providers: HashMap<String, ProviderConfig>,
    pub routing: RoutingConfig,
    pub cache: CacheConfig,
    pub rate_limit: RateLimitConfig,
}

#[derive(Debug, Clone, Default)]
pub struct AuthConfig {
    pub api_keys: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub kind: ProviderKind,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub models: Vec<String>,
    pub priority: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProviderKind {
    Openai,
    Anthropic,
    Ollama,
    Gemini,
    Mistral,
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
];

fn parse_provider_kind(s: &str) -> Option<ProviderKind> {
    match s.to_lowercase().as_str() {
        "openai" => Some(ProviderKind::Openai),
        "anthropic" => Some(ProviderKind::Anthropic),
        "gemini" => Some(ProviderKind::Gemini),
        "mistral" => Some(ProviderKind::Mistral),
        "ollama" => Some(ProviderKind::Ollama),
        _ => None,
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
                .filter(|s| !s.is_empty());
            let base_url = env::var(format!("{}_BASE_URL", prefix))
                .ok()
                .filter(|s| !s.is_empty());

            // Ollama doesn't need an API key, just check for base_url or models
            let models_str = env::var(format!("{}_MODELS", prefix))
                .ok()
                .filter(|s| !s.is_empty());
            let has_config = api_key.is_some()
                || (name == "ollama" && (base_url.is_some() || models_str.is_some()));

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

        // Second: prefix-based matching (e.g., "claude-" -> anthropic, "gpt-" -> openai)
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
