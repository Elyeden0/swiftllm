use std::sync::Arc;
use tracing::warn;

use crate::providers::types::{ChatRequest, ChatResponse};
use crate::providers::{Provider, ProviderError};

/// Attempt a request against a chain of providers, failing over on errors
pub async fn chat_with_failover(
    providers: &[(&str, Arc<dyn Provider>)],
    request: &ChatRequest,
) -> Result<(String, ChatResponse), ProviderError> {
    let mut last_error = None;

    for (name, provider) in providers {
        match provider.chat(request).await {
            Ok(response) => {
                return Ok((name.to_string(), response));
            }
            Err(e) => {
                warn!(
                    provider = *name,
                    error = %e,
                    "Provider failed, trying next in failover chain"
                );
                last_error = Some(e);
            }
        }
    }

    Err(last_error.unwrap_or_else(|| {
        ProviderError::Config("No providers available in failover chain".to_string())
    }))
}
