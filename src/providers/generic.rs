use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use reqwest::Client;
use tracing::{debug, error};

use super::schema::{AuthStyle, ProviderSchema};
use super::types::{ChatRequest, ChatResponse, EmbeddingRequest, EmbeddingResponse, StreamChunk};
use super::{Provider, ProviderError};

/// A provider implementation driven by a [`ProviderSchema`].
///
/// For any provider whose `ApiFormat` is `OpenAiCompatible`, this struct
/// reuses the standard OpenAI request/response format with just a different
/// base URL and authentication header.
pub struct GenericProvider {
    client: Client,
    schema: &'static ProviderSchema,
    api_key: String,
    base_url: String,
}

impl GenericProvider {
    /// Create a new generic provider.
    ///
    /// If `base_url_override` is `Some`, it takes precedence over the schema's
    /// default.
    pub fn new(
        schema: &'static ProviderSchema,
        api_key: String,
        base_url_override: Option<String>,
    ) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url_override.unwrap_or_else(|| schema.default_base_url.to_string()),
            api_key,
            schema,
        }
    }

    /// Build a request with the appropriate authentication.
    fn authed_post(&self, url: &str) -> reqwest::RequestBuilder {
        let mut builder = self
            .client
            .post(url)
            .header("Content-Type", "application/json");

        match &self.schema.auth_style {
            AuthStyle::Bearer => {
                builder = builder.header("Authorization", format!("Bearer {}", self.api_key));
            }
            AuthStyle::Header(name) => {
                builder = builder.header(*name, &self.api_key);
            }
            AuthStyle::Query(param) => {
                builder = builder.query(&[(*param, &self.api_key)]);
            }
            AuthStyle::None => {}
        }

        builder
    }
}

#[async_trait]
impl Provider for GenericProvider {
    fn name(&self) -> &str {
        self.schema.name
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        debug!(
            "{} chat request for model: {}",
            self.schema.name, request.model
        );

        let url = format!("{}/chat/completions", self.base_url);

        let mut req = ChatRequest::clone(request);
        req.stream = Some(false);

        let response = self
            .authed_post(&url)
            .json(&req)
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        let status = response.status().as_u16();
        if status >= 400 {
            let body = response.text().await.unwrap_or_default();
            error!("{} API error {}: {}", self.schema.name, status, body);
            return Err(ProviderError::Api {
                status,
                message: body,
            });
        }

        response
            .json::<ChatResponse>()
            .await
            .map_err(|e| ProviderError::Parse(e.to_string()))
    }

    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, ProviderError>>, ProviderError> {
        if !self.schema.supports_streaming {
            return Err(ProviderError::Config(format!(
                "{} does not support streaming",
                self.schema.name
            )));
        }

        debug!(
            "{} streaming request for model: {}",
            self.schema.name, request.model
        );

        let url = format!("{}/chat/completions", self.base_url);

        let mut req = ChatRequest::clone(request);
        req.stream = Some(true);

        let response = self
            .authed_post(&url)
            .json(&req)
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        let status = response.status().as_u16();
        if status >= 400 {
            let body = response.text().await.unwrap_or_default();
            return Err(ProviderError::Api {
                status,
                message: body,
            });
        }

        let stream = response
            .bytes_stream()
            .map(|result| match result {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    parse_sse_chunks(&text)
                }
                Err(e) => vec![Err(ProviderError::Network(e.to_string()))],
            })
            .flat_map(futures::stream::iter);

        Ok(Box::pin(stream))
    }

    async fn embeddings(
        &self,
        request: &EmbeddingRequest,
    ) -> Result<EmbeddingResponse, ProviderError> {
        debug!(
            "{} embeddings request for model: {}",
            self.schema.name, request.model
        );

        let url = format!("{}/embeddings", self.base_url);

        let response = self
            .authed_post(&url)
            .json(request)
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        let status = response.status().as_u16();
        if status >= 400 {
            let body = response.text().await.unwrap_or_default();
            error!(
                "{} embeddings API error {}: {}",
                self.schema.name, status, body
            );
            return Err(ProviderError::Api {
                status,
                message: body,
            });
        }

        response
            .json::<EmbeddingResponse>()
            .await
            .map_err(|e| ProviderError::Parse(e.to_string()))
    }
}

/// Parse SSE data lines into StreamChunk objects (same logic as the OpenAI provider).
fn parse_sse_chunks(text: &str) -> Vec<Result<StreamChunk, ProviderError>> {
    let mut chunks = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                continue;
            }
            match serde_json::from_str::<StreamChunk>(data) {
                Ok(chunk) => chunks.push(Ok(chunk)),
                Err(e) => {
                    debug!("Failed to parse SSE chunk: {} — data: {}", e, data);
                }
            }
        }
    }

    chunks
}

#[cfg(test)]
mod tests {
    use crate::providers::schema::{ApiFormat, AuthStyle, ProviderSchema};

    static TEST_SCHEMA: ProviderSchema = ProviderSchema {
        name: "test-generic",
        default_base_url: "https://api.test.example.com/v1",
        auth_style: AuthStyle::Bearer,
        format: ApiFormat::OpenAiCompatible,
        known_models: &["test-model"],
        supports_streaming: true,
        supports_tools: false,
        supports_vision: false,
    };

    #[test]
    fn test_schema_name() {
        assert_eq!(TEST_SCHEMA.name, "test-generic");
    }

    #[test]
    fn test_schema_base_url_override_logic() {
        let base_url_override: Option<String> = Some("https://custom.example.com".to_string());
        let resolved =
            base_url_override.unwrap_or_else(|| TEST_SCHEMA.default_base_url.to_string());
        assert_eq!(resolved, "https://custom.example.com");
    }

    #[test]
    fn test_schema_default_base_url_logic() {
        let base_url_override: Option<String> = None;
        let resolved =
            base_url_override.unwrap_or_else(|| TEST_SCHEMA.default_base_url.to_string());
        assert_eq!(resolved, "https://api.test.example.com/v1");
    }
}
