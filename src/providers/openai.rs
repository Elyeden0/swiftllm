use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::Client;
use tracing::{debug, error};

use super::types::{ChatRequest, ChatResponse, StreamChunk};
use super::{Provider, ProviderError};

pub struct OpenAiProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

impl OpenAiProvider {
    pub fn new(api_key: String, base_url: Option<String>) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url: base_url.unwrap_or_else(|| "https://api.openai.com".to_string()),
        }
    }
}

#[async_trait]
impl Provider for OpenAiProvider {
    fn name(&self) -> &str {
        "openai"
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        debug!("OpenAI chat request for model: {}", request.model);

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
            error!("OpenAI API error {}: {}", status, body);
            return Err(ProviderError::Api { status, message: body });
        }

        response
            .json::<ChatResponse>()
            .await
            .map_err(|e| ProviderError::Parse(e.to_string()))
    }

    async fn chat_stream(
        &self,
        _request: &ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, ProviderError>>, ProviderError> {
        // TODO: implement SSE streaming
        Err(ProviderError::Network("Streaming not yet implemented".to_string()))
    }
}
