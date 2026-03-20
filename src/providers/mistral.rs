use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use reqwest::Client;
use tracing::{debug, error};

use super::types::{ChatRequest, ChatResponse, StreamChunk};
use super::{Provider, ProviderError};

/// Mistral AI provider — uses an OpenAI-compatible API format
pub struct MistralProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

impl MistralProvider {
    pub fn new(api_key: String, base_url: Option<String>) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url: base_url.unwrap_or_else(|| "https://api.mistral.ai".to_string()),
        }
    }
}

#[async_trait]
impl Provider for MistralProvider {
    fn name(&self) -> &str {
        "mistral"
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        debug!("Mistral chat request for model: {}", request.model);

        let url = format!("{}/v1/chat/completions", self.base_url);

        let mut req = ChatRequest::clone(request);
        req.stream = Some(false);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&req)
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        let status = response.status().as_u16();
        if status >= 400 {
            let body = response.text().await.unwrap_or_default();
            error!("Mistral API error {}: {}", status, body);
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
        debug!("Mistral streaming request for model: {}", request.model);

        let url = format!("{}/v1/chat/completions", self.base_url);

        let mut req = ChatRequest::clone(request);
        req.stream = Some(true);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
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
}

/// Parse SSE data lines into StreamChunk objects (same format as OpenAI)
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
                    debug!("Failed to parse Mistral SSE chunk: {} — data: {}", e, data);
                }
            }
        }
    }

    chunks
}
