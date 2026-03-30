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
use crate::middleware::rate_limit::RateLimiter;
use crate::providers::anthropic::AnthropicProvider;
use crate::providers::bedrock::BedrockProvider;
use crate::providers::gemini::GeminiProvider;
use crate::providers::groq::GroqProvider;
use crate::providers::mistral::MistralProvider;
use crate::providers::ollama::OllamaProvider;
use crate::providers::openai::OpenAiProvider;
use crate::providers::together::TogetherProvider;
use crate::providers::types::{ChatRequest, EmbeddingRequest};
use crate::providers::{Provider, ProviderError};
use secrecy::ExposeSecret;

/// Shared application state
pub struct AppState {
    pub config: Config,
    pub providers: HashMap<String, Arc<dyn Provider>>,
    pub cache: Option<ResponseCache>,
    pub cost_tracker: CostTracker,
    pub rate_limiter: Option<RateLimiter>,
}

impl AppState {
    pub fn new(config: Config) -> Self {
        let mut providers: HashMap<String, Arc<dyn Provider>> = HashMap::new();

        for (name, provider_config) in &config.providers {
            let provider: Arc<dyn Provider> = match provider_config.kind {
                ProviderKind::Openai => Arc::new(OpenAiProvider::new(
                    provider_config
                        .api_key
                        .as_ref()
                        .map(|s| s.expose_secret().to_string())
                        .unwrap_or_default(),
                    provider_config.base_url.clone(),
                )),
                ProviderKind::Anthropic => Arc::new(AnthropicProvider::new(
                    provider_config
                        .api_key
                        .as_ref()
                        .map(|s| s.expose_secret().to_string())
                        .unwrap_or_default(),
                    provider_config.base_url.clone(),
                )),
                ProviderKind::Gemini => Arc::new(GeminiProvider::new(
                    provider_config
                        .api_key
                        .as_ref()
                        .map(|s| s.expose_secret().to_string())
                        .unwrap_or_default(),
                    provider_config.base_url.clone(),
                )),
                ProviderKind::Mistral => Arc::new(MistralProvider::new(
                    provider_config
                        .api_key
                        .as_ref()
                        .map(|s| s.expose_secret().to_string())
                        .unwrap_or_default(),
                    provider_config.base_url.clone(),
                )),
                ProviderKind::Ollama => {
                    Arc::new(OllamaProvider::new(provider_config.base_url.clone()))
                }
                ProviderKind::Groq => Arc::new(GroqProvider::new(
                    provider_config
                        .api_key
                        .as_ref()
                        .map(|s| s.expose_secret().to_string())
                        .unwrap_or_default(),
                    provider_config.base_url.clone(),
                )),
                ProviderKind::Together => Arc::new(TogetherProvider::new(
                    provider_config
                        .api_key
                        .as_ref()
                        .map(|s| s.expose_secret().to_string())
                        .unwrap_or_default(),
                    provider_config.base_url.clone(),
                )),
                ProviderKind::Bedrock => {
                    let region = std::env::var("BEDROCK_REGION")
                        .or_else(|_| std::env::var("AWS_REGION"))
                        .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
                        .unwrap_or_else(|_| "us-east-1".to_string());
                    let access_key = std::env::var("AWS_ACCESS_KEY_ID").unwrap_or_default();
                    let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY").unwrap_or_default();
                    let session_token = std::env::var("AWS_SESSION_TOKEN").ok();
                    Arc::new(BedrockProvider::new(
                        region,
                        access_key,
                        secret_key,
                        session_token,
                    ))
                }
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

        let rate_limiter = if config.rate_limit.enabled {
            let default_limit = Some(crate::middleware::rate_limit::RateLimitConfig {
                max_requests: config.rate_limit.max_requests,
                window: std::time::Duration::from_secs(config.rate_limit.window_seconds),
            });

            let mut provider_limits = HashMap::new();
            for (name, limit) in &config.rate_limit.providers {
                provider_limits.insert(
                    name.clone(),
                    crate::middleware::rate_limit::RateLimitConfig {
                        max_requests: limit.max_requests,
                        window: std::time::Duration::from_secs(limit.window_seconds),
                    },
                );
            }

            Some(RateLimiter::new(provider_limits, default_limit))
        } else {
            None
        };

        Self {
            config,
            providers,
            cache,
            cost_tracker: CostTracker::new(),
            rate_limiter,
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
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/models", get(list_models))
        .route("/health", get(health_check))
        .route("/api/stats", get(get_stats))
        .route("/dashboard", get(dashboard))
        .route("/dashboard/styles.css", get(dashboard_css))
        .route("/dashboard/app.js", get(dashboard_js))
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

    if let Some(limiter) = &state.rate_limiter {
        stats["rate_limits"] = serde_json::to_value(limiter.stats()).unwrap();
    }

    Json(stats)
}

/// Embedded dashboard HTML
async fn dashboard() -> impl IntoResponse {
    Html(include_str!("../dashboard/index.html"))
}

/// Embedded dashboard CSS
async fn dashboard_css() -> impl IntoResponse {
    Response::builder()
        .header("Content-Type", "text/css")
        .body(include_str!("../dashboard/styles.css").to_string())
        .unwrap()
}

/// Embedded dashboard JS
async fn dashboard_js() -> impl IntoResponse {
    Response::builder()
        .header("Content-Type", "application/javascript")
        .body(include_str!("../dashboard/app.js").to_string())
        .unwrap()
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
            Some(key)
                if state
                    .config
                    .auth
                    .api_keys
                    .iter()
                    .any(|k| k.expose_secret() == key) => {}
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

    // Find provider for the requested model (needed for rate limit check)
    let (provider_name, _provider_config) = state
        .config
        .find_provider_for_model(&request.model)
        .ok_or_else(|| {
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

    // Rate limit check (per provider)
    if let Some(limiter) = &state.rate_limiter {
        if let Err(retry_after) = limiter.check(provider_name) {
            warn!(
                provider = provider_name.as_str(),
                retry_after = format!("{:.1}s", retry_after).as_str(),
                "Rate limited"
            );
            return Err((
                StatusCode::TOO_MANY_REQUESTS,
                Json(serde_json::json!({
                    "error": {
                        "message": format!(
                            "Rate limit exceeded for provider '{}'. Retry after {:.1}s",
                            provider_name, retry_after
                        ),
                        "type": "rate_limit_error",
                        "retry_after": retry_after,
                    }
                })),
            ));
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

/// Embeddings endpoint — routes to the appropriate provider
async fn embeddings(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<EmbeddingRequest>,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
    // Auth check
    if !state.config.auth.api_keys.is_empty() {
        let provided_key = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));

        match provided_key {
            Some(key)
                if state
                    .config
                    .auth
                    .api_keys
                    .iter()
                    .any(|k| k.expose_secret() == key) => {}
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

    // Find provider for the requested model
    let (provider_name, _provider_config) = state
        .config
        .find_provider_for_model(&request.model)
        .ok_or_else(|| {
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

    // Rate limit check
    if let Some(limiter) = &state.rate_limiter {
        if let Err(retry_after) = limiter.check(provider_name) {
            warn!(
                provider = provider_name.as_str(),
                retry_after = format!("{:.1}s", retry_after).as_str(),
                "Rate limited"
            );
            return Err((
                StatusCode::TOO_MANY_REQUESTS,
                Json(serde_json::json!({
                    "error": {
                        "message": format!(
                            "Rate limit exceeded for provider '{}'. Retry after {:.1}s",
                            provider_name, retry_after
                        ),
                        "type": "rate_limit_error",
                        "retry_after": retry_after,
                    }
                })),
            ));
        }
    }

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
        "Routing embedding request"
    );

    match provider.embeddings(&request).await {
        Ok(response) => {
            // Track cost (embeddings only have prompt tokens)
            state.cost_tracker.record_request(
                provider_name,
                &request.model,
                response.usage.prompt_tokens,
                0,
            );
            Ok(Json(response).into_response())
        }
        Err(e) => {
            state.cost_tracker.record_error(provider_name);
            Err(provider_error_to_response(e))
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

fn provider_error_to_response(err: ProviderError) -> (StatusCode, Json<serde_json::Value>) {
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
