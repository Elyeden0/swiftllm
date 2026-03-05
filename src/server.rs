use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures::StreamExt;
use std::collections::HashMap;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, info, warn};

use crate::config::{Config, ProviderKind};
use crate::middleware::cache::ResponseCache;
use crate::middleware::cost::CostTracker;
use crate::providers::anthropic::AnthropicProvider;
use crate::providers::ollama::OllamaProvider;
use crate::providers::openai::OpenAiProvider;
use crate::providers::types::ChatRequest;
use crate::providers::{Provider, ProviderError};

/// Shared application state
pub struct AppState {
    pub config: Config,
    pub providers: HashMap<String, Arc<dyn Provider>>,
    pub cache: Option<ResponseCache>,
    pub cost_tracker: CostTracker,
}

impl AppState {
    pub fn new(config: Config) -> Self {
        let mut providers: HashMap<String, Arc<dyn Provider>> = HashMap::new();

        for (name, provider_config) in &config.providers {
            let provider: Arc<dyn Provider> = match provider_config.kind {
                ProviderKind::Openai => Arc::new(OpenAiProvider::new(
                    provider_config.api_key.clone().unwrap_or_default(),
                    provider_config.base_url.clone(),
                )),
                ProviderKind::Anthropic => Arc::new(AnthropicProvider::new(
                    provider_config.api_key.clone().unwrap_or_default(),
                    provider_config.base_url.clone(),
                )),
                ProviderKind::Ollama => Arc::new(OllamaProvider::new(
                    provider_config.base_url.clone(),
                )),
            };
            providers.insert(name.clone(), provider);
        }

        let cache = if config.cache.enabled {
            Some(ResponseCache::new(
                config.cache.max_size,
                config.cache.ttl_seconds,
            ))
        } else {
            None
        };

        Self {
            config,
            providers,
            cache,
            cost_tracker: CostTracker::new(),
        }
    }

    /// Get providers sorted by priority for failover
    fn failover_providers(&self) -> Vec<(&str, Arc<dyn Provider>)> {
        let sorted: Vec<_> = self.config.providers_by_priority();
        sorted
            .iter()
            .filter_map(|(name, _)| {
                self.providers
                    .get(name.as_str())
                    .map(|p| (name.as_str(), p.clone()))
            })
            .collect()
    }
}

/// Build the axum router
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/health", get(health_check))
        .route("/api/stats", get(get_stats))
        .route("/dashboard", get(dashboard))
        .with_state(state)
}

/// Health check endpoint
async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

/// Stats API endpoint
async fn get_stats(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut stats = serde_json::to_value(state.cost_tracker.snapshot()).unwrap();

    if let Some(cache) = &state.cache {
        stats["cache"] = serde_json::to_value(cache.stats()).unwrap();
    }

    Json(stats)
}

/// Embedded dashboard
async fn dashboard() -> impl IntoResponse {
    Html(include_str!("../dashboard/index.html"))
}

/// List available models (aggregated from all providers)
async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut models = Vec::new();

    for (name, provider_config) in &state.config.providers {
        for model in &provider_config.models {
            models.push(serde_json::json!({
                "id": model,
                "object": "model",
                "owned_by": name,
            }));
        }
    }

    Json(serde_json::json!({
        "object": "list",
        "data": models,
    }))
}

/// Main chat completions endpoint — routes to the appropriate provider
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<ChatRequest>,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
    // Auth check
    if !state.config.auth.api_keys.is_empty() {
        let provided_key = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));

        match provided_key {
            Some(key) if state.config.auth.api_keys.contains(&key.to_string()) => {}
            _ => {
                return Err((
                    StatusCode::UNAUTHORIZED,
                    Json(serde_json::json!({
                        "error": {
                            "message": "Invalid API key",
                            "type": "authentication_error",
                        }
                    })),
                ));
            }
        }
    }

    // Check cache for non-streaming requests
    if !request.is_streaming() {
        if let Some(cache) = &state.cache {
            if let Some(cached) = cache.get(&request) {
                info!(model = %request.model, "Cache hit");
                state.cost_tracker.record_cache_hit();
                return Ok(Json(cached).into_response());
            }
        }
    }

    // Find provider for the requested model
    let (provider_name, _provider_config) =
        state.config.find_provider_for_model(&request.model).ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("No provider configured for model: {}", request.model),
                        "type": "invalid_request_error",
                    }
                })),
            )
        })?;

    let provider = state.providers.get(provider_name).ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": {
                    "message": "Provider not initialized",
                    "type": "server_error",
                }
            })),
        )
    })?;

    info!(
        model = %request.model,
        provider = provider_name.as_str(),
        stream = request.is_streaming(),
        "Routing request"
    );

    if request.is_streaming() {
        handle_streaming(provider.clone(), request)
            .await
            .map_err(|e| {
                state.cost_tracker.record_error(provider_name);
                provider_error_to_response(e)
            })
    } else {
        // Try primary provider, failover on error
        match provider.chat(&request).await {
            Ok(response) => {
                // Track cost
                if let Some(usage) = &response.usage {
                    state.cost_tracker.record_request(
                        provider_name,
                        &request.model,
                        usage.prompt_tokens,
                        usage.completion_tokens,
                    );
                }

                // Cache the response
                if let Some(cache) = &state.cache {
                    cache.put(&request, response.clone());
                }

                Ok(Json(response).into_response())
            }
            Err(primary_err) => {
                warn!(
                    provider = provider_name.as_str(),
                    error = %primary_err,
                    "Primary provider failed, attempting failover"
                );
                state.cost_tracker.record_error(provider_name);

                // Try failover chain (skip the provider that already failed)
                let failover_chain: Vec<_> = state
                    .failover_providers()
                    .into_iter()
                    .filter(|(name, _)| *name != provider_name.as_str())
                    .collect();

                if failover_chain.is_empty() {
                    return Err(provider_error_to_response(primary_err));
                }

                match crate::failover::chat_with_failover(&failover_chain, &request).await {
                    Ok((fallback_name, response)) => {
                        info!(
                            fallback_provider = fallback_name.as_str(),
                            "Failover succeeded"
                        );
                        if let Some(usage) = &response.usage {
                            state.cost_tracker.record_request(
                                &fallback_name,
                                &request.model,
                                usage.prompt_tokens,
                                usage.completion_tokens,
                            );
                        }
                        if let Some(cache) = &state.cache {
                            cache.put(&request, response.clone());
                        }
                        Ok(Json(response).into_response())
                    }
                    Err(e) => Err(provider_error_to_response(e)),
                }
            }
        }
    }
}

async fn handle_streaming(
    provider: Arc<dyn Provider>,
    request: ChatRequest,
) -> Result<Response, ProviderError> {
    let mut stream = provider.chat_stream(&request).await?;

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, std::convert::Infallible>>(32);

    tokio::spawn(async move {
        while let Some(result) = stream.next().await {
            match result {
                Ok(chunk) => {
                    let data = format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap());
                    if tx.send(Ok(data)).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    error!("Stream error: {}", e);
                    break;
                }
            }
        }
        let _ = tx.send(Ok("data: [DONE]\n\n".to_string())).await;
    });

    let body = Body::from_stream(ReceiverStream::new(rx));

    Ok(Response::builder()
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("Connection", "keep-alive")
        .body(body)
        .unwrap())
}

fn provider_error_to_response(
    err: ProviderError,
) -> (StatusCode, Json<serde_json::Value>) {
    let (status, message) = match &err {
        ProviderError::Network(msg) => (StatusCode::BAD_GATEWAY, msg.clone()),
        ProviderError::Api { status, message } => (
            StatusCode::from_u16(*status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            message.clone(),
        ),
        ProviderError::Parse(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
        ProviderError::Config(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
    };

    (
        status,
        Json(serde_json::json!({
            "error": {
                "message": message,
                "type": "proxy_error",
            }
        })),
    )
}
