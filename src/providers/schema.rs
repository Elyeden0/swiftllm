/// Declarative provider definition.
///
/// Most LLM providers are OpenAI-compatible — they accept the same request
/// format at a different base URL. Instead of writing custom modules for each,
/// define them declaratively with a `ProviderSchema`.
#[derive(Debug, Clone)]
pub struct ProviderSchema {
    pub name: &'static str,
    pub default_base_url: &'static str,
    pub auth_style: AuthStyle,
    pub format: ApiFormat,
    pub known_models: &'static [&'static str],
    pub supports_streaming: bool,
    pub supports_tools: bool,
    pub supports_vision: bool,
}

/// How the provider expects authentication credentials.
#[derive(Debug, Clone, PartialEq)]
pub enum AuthStyle {
    /// Standard `Authorization: Bearer <key>` header.
    Bearer,
    /// Custom header name, e.g. `x-api-key`.
    Header(&'static str),
    /// API key sent as a query parameter.
    Query(&'static str),
    /// No authentication required (e.g. local models).
    None,
}

/// The request/response wire format the provider uses.
#[derive(Debug, Clone, PartialEq)]
pub enum ApiFormat {
    /// Accepts the standard OpenAI chat-completions request format.
    OpenAiCompatible,
    /// Requires a hand-written provider module (Anthropic, Gemini, Bedrock, …).
    Custom,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_construction() {
        let schema = ProviderSchema {
            name: "test-provider",
            default_base_url: "https://api.test.com/v1",
            auth_style: AuthStyle::Bearer,
            format: ApiFormat::OpenAiCompatible,
            known_models: &["model-a", "model-b"],
            supports_streaming: true,
            supports_tools: false,
            supports_vision: false,
        };
        assert_eq!(schema.name, "test-provider");
        assert_eq!(schema.known_models.len(), 2);
        assert!(schema.supports_streaming);
    }

    #[test]
    fn test_auth_style_variants() {
        assert_eq!(AuthStyle::Bearer, AuthStyle::Bearer);
        assert_eq!(
            AuthStyle::Header("x-api-key"),
            AuthStyle::Header("x-api-key")
        );
        assert_ne!(AuthStyle::Bearer, AuthStyle::None);
    }

    #[test]
    fn test_api_format_variants() {
        assert_eq!(ApiFormat::OpenAiCompatible, ApiFormat::OpenAiCompatible);
        assert_ne!(ApiFormat::OpenAiCompatible, ApiFormat::Custom);
    }
}
