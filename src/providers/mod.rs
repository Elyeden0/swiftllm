pub mod anthropic;
pub mod ollama;
pub mod openai;
pub mod types;

use async_trait::async_trait;
use futures::stream::BoxStream;
use types::{ChatRequest, ChatResponse, StreamChunk};

/// Error type for provider operations
#[derive(Debug)]
#[allow(dead_code)]
pub enum ProviderError {
    /// HTTP or network error
    Network(String),
    /// Provider returned an error response
    Api { status: u16, message: String },
    /// Failed to parse response
    Parse(String),
    /// Configuration error
    Config(String),
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderError::Network(msg) => write!(f, "Network error: {}", msg),
            ProviderError::Api { status, message } => {
                write!(f, "API error ({}): {}", status, message)
            }
            ProviderError::Parse(msg) => write!(f, "Parse error: {}", msg),
            ProviderError::Config(msg) => write!(f, "Config error: {}", msg),
        }
    }
}

impl std::error::Error for ProviderError {}

/// Trait that all LLM providers must implement
#[async_trait]
pub trait Provider: Send + Sync {
    /// Provider name for logging
    fn name(&self) -> &str;

    /// Send a non-streaming chat completion request
    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ProviderError>;

    /// Send a streaming chat completion request
    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, ProviderError>>, ProviderError>;
}
