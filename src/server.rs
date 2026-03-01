use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures::StreamExt;
use std::collections::HashMap;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, info};

use crate::config::{Config, ProviderKind};
use crate::providers::types::ChatRequest;
use crate::providers::{Provider, ProviderError};
use crate::providers::anthropic::AnthropicProvider;
use crate::providers::ollama::OllamaProvider;
use crate::providers::openai::OpenAiProvider;

pub struct AppState {
    pub config: Config,
    pub providers: HashMap<String, Arc<dyn Provider>>,
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
        Self { config, providers }
    }
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/health", get(health_check))
        // TODO: caching middleware
        // TODO: cost tracking
        .with_state(state)
}

async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok", "version": env!("CARGO_PKG_VERSION") }))
}

async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut models = Vec::new();
    for (name, provider_config) in &state.config.providers {
        for model in &provider_config.models {
            models.push(serde_json::json!({ "id": model, "object": "model", "owned_by": name }));
        }
    }
    Json(serde_json::json!({ "object": "list", "data": models }))
}

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
                return Err((StatusCode::UNAUTHORIZED, Json(serde_json::json!({
                    "error": { "message": "Invalid API key", "type": "authentication_error" }
                }))));
            }
        }
    }

    let (provider_name, _) =
        state.config.find_provider_for_model(&request.model).ok_or_else(|| {
            (StatusCode::BAD_REQUEST, Json(serde_json::json!({
                "error": { "message": format!("No provider for model: {}", request.model), "type": "invalid_request_error" }
            })))
        })?;

    let provider = state.providers.get(provider_name).ok_or_else(|| {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
            "error": { "message": "Provider not initialized", "type": "server_error" }
        })))
    })?;

    info!(model = %request.model, provider = provider_name.as_str(), stream = request.is_streaming(), "Routing request");

    if request.is_streaming() {
        handle_streaming(provider.clone(), request).await.map_err(provider_error_to_response)
    } else {
        let response = provider.chat(&request).await.map_err(provider_error_to_response)?;
        Ok(Json(response).into_response())
    }
}

async fn handle_streaming(provider: Arc<dyn Provider>, request: ChatRequest) -> Result<Response, ProviderError> {
    let mut stream = provider.chat_stream(&request).await?;
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, std::convert::Infallible>>(32);

    tokio::spawn(async move {
        while let Some(result) = stream.next().await {
            match result {
                Ok(chunk) => {
                    let data = format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap());
                    if tx.send(Ok(data)).await.is_err() { break; }
                }
                Err(e) => { error!("Stream error: {}", e); break; }
            }
        }
        let _ = tx.send(Ok("data: [DONE]\n\n".to_string())).await;
    });

    let body = Body::from_stream(ReceiverStream::new(rx));
    Ok(Response::builder()
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("Connection", "keep-alive")
        .body(body).unwrap())
}

fn provider_error_to_response(err: ProviderError) -> (StatusCode, Json<serde_json::Value>) {
    let (status, message) = match &err {
        ProviderError::Network(msg) => (StatusCode::BAD_GATEWAY, msg.clone()),
        ProviderError::Api { status, message } => (
            StatusCode::from_u16(*status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), message.clone(),
        ),
        ProviderError::Parse(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
    };
    (status, Json(serde_json::json!({ "error": { "message": message, "type": "proxy_error" } })))
}
