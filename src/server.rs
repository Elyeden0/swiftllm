use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::info;

use crate::config::{Config, ProviderKind};
use crate::providers::types::ChatRequest;
use crate::providers::{Provider, ProviderError};
use crate::providers::openai::OpenAiProvider;

pub struct AppState {
    pub config: Config,
    pub providers: HashMap<String, Arc<dyn Provider>>,
}

impl AppState {
    pub fn new(config: Config) -> Self {
        let mut providers: HashMap<String, Arc<dyn Provider>> = HashMap::new();

        for (name, provider_config) in &config.providers {
            let provider: Option<Arc<dyn Provider>> = match provider_config.kind {
                ProviderKind::Openai => Some(Arc::new(OpenAiProvider::new(
                    provider_config.api_key.clone().unwrap_or_default(),
                    provider_config.base_url.clone(),
                ))),
                // TODO: Anthropic provider
                // TODO: Ollama provider
                _ => {
                    tracing::warn!("Provider kind {:?} not yet implemented", provider_config.kind);
                    None
                }
            };
            if let Some(p) = provider {
                providers.insert(name.clone(), p);
            }
        }

        Self { config, providers }
    }
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/health", get(health_check))
        // TODO: add /v1/models endpoint
        // TODO: add auth middleware
        .with_state(state)
}

async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let (provider_name, _) =
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
                "error": { "message": "Provider not initialized", "type": "server_error" }
            })),
        )
    })?;

    info!(model = %request.model, provider = provider_name.as_str(), "Routing request");

    // TODO: handle streaming requests
    // TODO: add response caching
    let response = provider.chat(&request).await.map_err(|e| {
        let (status, message) = match &e {
            ProviderError::Network(msg) => (StatusCode::BAD_GATEWAY, msg.clone()),
            ProviderError::Api { status, message } => (
                StatusCode::from_u16(*status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                message.clone(),
            ),
            ProviderError::Parse(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
        };
        (status, Json(serde_json::json!({ "error": { "message": message, "type": "proxy_error" } })))
    })?;

    Ok(Json(serde_json::to_value(response).unwrap()))
}
