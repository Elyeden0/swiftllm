//! Image generation endpoint (OpenAI-compatible).

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, instrument};

use crate::server::AppState;
use secrecy::ExposeSecret;

// ── Types ──────────────────────────────────────────────────────────────────

/// OpenAI-compatible image generation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_n")]
    pub n: u32,
    #[serde(default = "default_size")]
    pub size: String,
    #[serde(default = "default_quality")]
    pub quality: String,
    #[serde(default = "default_response_format")]
    pub response_format: String,
}

fn default_n() -> u32 {
    1
}
fn default_size() -> String {
    "1024x1024".to_string()
}
fn default_quality() -> String {
    "standard".to_string()
}
fn default_response_format() -> String {
    "url".to_string()
}

/// A single generated image datum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
}

/// OpenAI-compatible image generation response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationResponse {
    pub created: u64,
    pub data: Vec<ImageData>,
}

// ── Stability AI translation ───────────────────────────────────────────────

/// Convert our unified request into a Stability AI request body.
fn to_stability_body(request: &ImageGenerationRequest) -> serde_json::Value {
    let (width, height) = parse_size(&request.size);
    serde_json::json!({
        "text_prompts": [{"text": request.prompt, "weight": 1.0}],
        "cfg_scale": 7,
        "samples": request.n,
        "width": width,
        "height": height,
    })
}

fn parse_size(size: &str) -> (u32, u32) {
    let parts: Vec<&str> = size.split('x').collect();
    if parts.len() == 2 {
        let w = parts[0].parse().unwrap_or(1024);
        let h = parts[1].parse().unwrap_or(1024);
        (w, h)
    } else {
        (1024, 1024)
    }
}

fn is_stability_model(model: &str) -> bool {
    let m = model.to_lowercase();
    m.starts_with("stable-diffusion") || m.starts_with("stability-") || m.starts_with("sd-")
}

// ── Handler ────────────────────────────────────────────────────────────────

/// POST /v1/images/generations — image generation.
#[instrument(
    name = "gen_ai.images.generation",
    skip_all,
    fields(gen_ai.request.model = %request.model)
)]
pub async fn generations(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<ImageGenerationRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    check_auth(&state, &headers)?;

    let (provider_name, _) = state
        .config
        .find_provider_for_model(&request.model)
        .ok_or_else(|| {
            api_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &format!("No provider configured for model: {}", request.model),
            )
        })?;

    let provider_config = state.config.providers.get(provider_name).ok_or_else(|| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "Provider not initialized",
        )
    })?;

    let api_key = provider_config
        .api_key
        .as_ref()
        .map(|s| s.expose_secret().to_string())
        .unwrap_or_default();

    info!(model = %request.model, provider = provider_name, "Routing image generation request");

    let client = reqwest::Client::new();

    let resp = if is_stability_model(&request.model) {
        // Stability AI path
        let base = provider_config
            .base_url
            .as_deref()
            .unwrap_or("https://api.stability.ai");
        let engine_id = &request.model;
        client
            .post(format!(
                "{}/v1/generation/{}/text-to-image",
                base.trim_end_matches('/'),
                engine_id
            ))
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&to_stability_body(&request))
            .send()
            .await
    } else {
        // OpenAI-compatible path (default)
        let base = provider_config
            .base_url
            .as_deref()
            .unwrap_or("https://api.openai.com/v1");
        client
            .post(format!("{}/images/generations", base.trim_end_matches('/')))
            .bearer_auth(&api_key)
            .json(&request)
            .send()
            .await
    }
    .map_err(|e| api_error(StatusCode::BAD_GATEWAY, "proxy_error", &e.to_string()))?;

    if !resp.status().is_success() {
        let status = StatusCode::from_u16(resp.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let body = resp.text().await.unwrap_or_default();
        return Err(api_error(status, "proxy_error", &body));
    }

    // For Stability AI, translate the response to OpenAI format
    if is_stability_model(&request.model) {
        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| api_error(StatusCode::BAD_GATEWAY, "proxy_error", &e.to_string()))?;

        let data: Vec<ImageData> = body["artifacts"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .map(|a| ImageData {
                url: None,
                b64_json: a["base64"].as_str().map(|s| s.to_string()),
                revised_prompt: None,
            })
            .collect();

        let response = ImageGenerationResponse {
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            data,
        };
        return Ok(Json(response));
    }

    // OpenAI-compatible: pass through
    let response: ImageGenerationResponse = resp
        .json()
        .await
        .map_err(|e| api_error(StatusCode::BAD_GATEWAY, "proxy_error", &e.to_string()))?;

    Ok(Json(response))
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn check_auth(
    state: &AppState,
    headers: &HeaderMap,
) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    if state.config.auth.api_keys.is_empty() {
        return Ok(());
    }
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
                .any(|k| k.expose_secret() == key) =>
        {
            Ok(())
        }
        _ => Err(api_error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "Invalid API key",
        )),
    }
}

fn api_error(
    status: StatusCode,
    error_type: &str,
    message: &str,
) -> (StatusCode, Json<serde_json::Value>) {
    (
        status,
        Json(serde_json::json!({
            "error": {
                "message": message,
                "type": error_type,
            }
        })),
    )
}
