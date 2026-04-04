//! Audio endpoints: text-to-speech and speech-to-text (transcription).

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::Response,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, instrument};

use crate::server::AppState;
use secrecy::ExposeSecret;

// ── Text-to-Speech ─────────────────────────────────────────────────────────

/// OpenAI-compatible text-to-speech request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechRequest {
    pub model: String,
    pub input: String,
    #[serde(default = "default_voice")]
    pub voice: String,
    #[serde(default = "default_audio_format")]
    pub response_format: String,
    #[serde(default = "default_speed")]
    pub speed: f64,
}

fn default_voice() -> String {
    "alloy".to_string()
}
fn default_audio_format() -> String {
    "mp3".to_string()
}
fn default_speed() -> f64 {
    1.0
}

/// OpenAI-compatible transcription response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResponse {
    pub text: String,
}

/// Multipart transcription request fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionRequest {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
}

fn content_type_for_format(format: &str) -> &'static str {
    match format {
        "mp3" => "audio/mpeg",
        "opus" => "audio/opus",
        "aac" => "audio/aac",
        "flac" => "audio/flac",
        "wav" => "audio/wav",
        "pcm" => "audio/pcm",
        _ => "audio/mpeg",
    }
}

/// POST /v1/audio/speech — text-to-speech.
///
/// Forwards the request to the provider associated with the requested model
/// (e.g. `tts-1` → OpenAI) and streams back the raw audio bytes.
#[instrument(name = "gen_ai.audio.speech", skip_all, fields(gen_ai.request.model = %request.model))]
pub async fn speech(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<SpeechRequest>,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
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

    let base_url = provider_config
        .base_url
        .as_deref()
        .unwrap_or("https://api.openai.com/v1");

    let api_key = provider_config
        .api_key
        .as_ref()
        .map(|s| s.expose_secret().to_string())
        .unwrap_or_default();

    info!(model = %request.model, provider = provider_name, "Routing TTS request");

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/audio/speech", base_url.trim_end_matches('/')))
        .bearer_auth(&api_key)
        .json(&request)
        .send()
        .await
        .map_err(|e| api_error(StatusCode::BAD_GATEWAY, "proxy_error", &e.to_string()))?;

    if !resp.status().is_success() {
        let status = StatusCode::from_u16(resp.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let body = resp.text().await.unwrap_or_default();
        return Err(api_error(status, "proxy_error", &body));
    }

    let content_type = content_type_for_format(&request.response_format);
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| api_error(StatusCode::BAD_GATEWAY, "proxy_error", &e.to_string()))?;

    Ok(Response::builder()
        .header("Content-Type", content_type)
        .body(Body::from(bytes))
        .unwrap())
}

/// POST /v1/audio/transcriptions — speech-to-text.
///
/// Accepts multipart/form-data with an audio file and forwards to the provider.
/// For simplicity we re-serialize the body and forward it as-is.
#[instrument(name = "gen_ai.audio.transcription", skip_all)]
pub async fn transcriptions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
    check_auth(&state, &headers)?;

    // The model field is embedded in the multipart form; we default to whisper-1
    // and forward to the OpenAI provider (the most common transcription backend).
    let model = "whisper-1";
    let (provider_name, _) = state.config.find_provider_for_model(model).ok_or_else(|| {
        api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            &format!("No provider configured for model: {}", model),
        )
    })?;

    let provider_config = state.config.providers.get(provider_name).ok_or_else(|| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "Provider not initialized",
        )
    })?;

    let base_url = provider_config
        .base_url
        .as_deref()
        .unwrap_or("https://api.openai.com/v1");

    let api_key = provider_config
        .api_key
        .as_ref()
        .map(|s| s.expose_secret().to_string())
        .unwrap_or_default();

    info!(
        model = model,
        provider = provider_name,
        "Routing transcription request"
    );

    // Forward the raw multipart body to the upstream provider.
    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("multipart/form-data")
        .to_string();

    let client = reqwest::Client::new();
    let resp = client
        .post(format!(
            "{}/audio/transcriptions",
            base_url.trim_end_matches('/')
        ))
        .bearer_auth(&api_key)
        .header("Content-Type", content_type)
        .body(body.to_vec())
        .send()
        .await
        .map_err(|e| api_error(StatusCode::BAD_GATEWAY, "proxy_error", &e.to_string()))?;

    if !resp.status().is_success() {
        let status = StatusCode::from_u16(resp.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let body = resp.text().await.unwrap_or_default();
        return Err(api_error(status, "proxy_error", &body));
    }

    let response_body = resp
        .text()
        .await
        .map_err(|e| api_error(StatusCode::BAD_GATEWAY, "proxy_error", &e.to_string()))?;

    Ok(Response::builder()
        .header("Content-Type", "application/json")
        .body(Body::from(response_body))
        .unwrap())
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
