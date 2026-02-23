pub mod types;
pub mod openai;

use async_trait::async_trait;
use futures::stream::BoxStream;
use types::{ChatRequest, ChatResponse, StreamChunk};

#[derive(Debug)]
pub enum ProviderError {
    Network(String),
    Api { status: u16, message: String },
    Parse(String),
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderError::Network(msg) => write!(f, "Network error: {}", msg),
            ProviderError::Api { status, message } => write!(f, "API error ({}): {}", status, message),
            ProviderError::Parse(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for ProviderError {}

#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ProviderError>;
    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, ProviderError>>, ProviderError>;
}
