//! Search endpoint — SwiftLLM's own addition (not in the OpenAI API).
//!
//! Provides a unified interface to web search providers: Brave, Tavily,
//! Serper, and SearXNG.

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

/// Search request body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_provider")]
    pub provider: String,
    #[serde(default = "default_max_results")]
    pub max_results: u32,
}

fn default_provider() -> String {
    "brave".to_string()
}
fn default_max_results() -> u32 {
    10
}

/// A single search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub source: String,
}

/// Normalized search response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub query: String,
    pub total_results: u32,
}

// ── Handler ────────────────────────────────────────────────────────────────

/// POST /v1/search — web search across multiple providers.
#[instrument(name = "gen_ai.search", skip_all, fields(search.provider = %request.provider))]
pub async fn search(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<SearchRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    check_auth(&state, &headers)?;

    info!(
        query = %request.query,
        provider = %request.provider,
        max_results = request.max_results,
        "Routing search request"
    );

    let response = match request.provider.to_lowercase().as_str() {
        "brave" => search_brave(&state, &request).await?,
        "tavily" => search_tavily(&state, &request).await?,
        "serper" => search_serper(&state, &request).await?,
        "searxng" => search_searxng(&state, &request).await?,
        other => {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &format!("Unsupported search provider: {}. Supported: brave, tavily, serper, searxng", other),
            ));
        }
    };

    Ok(Json(response))
}

// ── Provider implementations ───────────────────────────────────────────────

async fn search_brave(
    _state: &AppState,
    request: &SearchRequest,
) -> Result<SearchResponse, (StatusCode, Json<serde_json::Value>)> {
    let api_key = std::env::var("BRAVE_API_KEY")
        .map_err(|_| api_error(StatusCode::BAD_REQUEST, "config_error", "BRAVE_API_KEY not set"))?;

    let client = reqwest::Client::new();
    let resp = client
        .get("https://api.search.brave.com/res/v1/web/search")
        .header("X-Subscription-Token", &api_key)
        .header("Accept", "application/json")
        .query(&[
            ("q", request.query.as_str()),
            ("count", &request.max_results.to_string()),
        ])
        .send()
        .await
        .map_err(|e| api_error(StatusCode::BAD_GATEWAY, "proxy_error", &e.to_string()))?;

    if !resp.status().is_success() {
        let status = StatusCode::from_u16(resp.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let body = resp.text().await.unwrap_or_default();
        return Err(api_error(status, "proxy_error", &body));
    }

    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| api_error(StatusCode::BAD_GATEWAY, "proxy_error", &e.to_string()))?;

    let results: Vec<SearchResult> = body["web"]["results"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .take(request.max_results as usize)
        .map(|r| SearchResult {
            title: r["title"].as_str().unwrap_or("").to_string(),
            url: r["url"].as_str().unwrap_or("").to_string(),
            snippet: r["description"].as_str().unwrap_or("").to_string(),
            source: "brave".to_string(),
        })
        .collect();

    let total = results.len() as u32;
    Ok(SearchResponse {
        results,
        query: request.query.clone(),
        total_results: total,
    })
}

async fn search_tavily(
    _state: &AppState,
    request: &SearchRequest,
) -> Result<SearchResponse, (StatusCode, Json<serde_json::Value>)> {
    let api_key = std::env::var("TAVILY_API_KEY")
        .map_err(|_| api_error(StatusCode::BAD_REQUEST, "config_error", "TAVILY_API_KEY not set"))?;

    let client = reqwest::Client::new();
    let resp = client
        .post("https://api.tavily.com/search")
        .json(&serde_json::json!({
            "api_key": api_key,
            "query": request.query,
            "max_results": request.max_results,
        }))
        .send()
        .await
        .map_err(|e| api_error(StatusCode::BAD_GATEWAY, "proxy_error", &e.to_string()))?;

    if !resp.status().is_success() {
        let status = StatusCode::from_u16(resp.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let body = resp.text().await.unwrap_or_default();
        return Err(api_error(status, "proxy_error", &body));
    }

    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| api_error(StatusCode::BAD_GATEWAY, "proxy_error", &e.to_string()))?;

    let results: Vec<SearchResult> = body["results"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .take(request.max_results as usize)
        .map(|r| SearchResult {
            title: r["title"].as_str().unwrap_or("").to_string(),
            url: r["url"].as_str().unwrap_or("").to_string(),
            snippet: r["content"].as_str().unwrap_or("").to_string(),
            source: "tavily".to_string(),
        })
        .collect();

    let total = results.len() as u32;
    Ok(SearchResponse {
        results,
        query: request.query.clone(),
        total_results: total,
    })
}

async fn search_serper(
    _state: &AppState,
    request: &SearchRequest,
) -> Result<SearchResponse, (StatusCode, Json<serde_json::Value>)> {
    let api_key = std::env::var("SERPER_API_KEY")
        .map_err(|_| api_error(StatusCode::BAD_REQUEST, "config_error", "SERPER_API_KEY not set"))?;

    let client = reqwest::Client::new();
    let resp = client
        .post("https://google.serper.dev/search")
        .header("X-API-KEY", &api_key)
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({
            "q": request.query,
            "num": request.max_results,
        }))
        .send()
        .await
        .map_err(|e| api_error(StatusCode::BAD_GATEWAY, "proxy_error", &e.to_string()))?;

    if !resp.status().is_success() {
        let status = StatusCode::from_u16(resp.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let body = resp.text().await.unwrap_or_default();
        return Err(api_error(status, "proxy_error", &body));
    }

    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| api_error(StatusCode::BAD_GATEWAY, "proxy_error", &e.to_string()))?;

    let results: Vec<SearchResult> = body["organic"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .take(request.max_results as usize)
        .map(|r| SearchResult {
            title: r["title"].as_str().unwrap_or("").to_string(),
            url: r["link"].as_str().unwrap_or("").to_string(),
            snippet: r["snippet"].as_str().unwrap_or("").to_string(),
            source: "serper".to_string(),
        })
        .collect();

    let total = results.len() as u32;
    Ok(SearchResponse {
        results,
        query: request.query.clone(),
        total_results: total,
    })
}

async fn search_searxng(
    _state: &AppState,
    request: &SearchRequest,
) -> Result<SearchResponse, (StatusCode, Json<serde_json::Value>)> {
    let base_url = std::env::var("SEARXNG_BASE_URL").map_err(|_| {
        api_error(
            StatusCode::BAD_REQUEST,
            "config_error",
            "SEARXNG_BASE_URL not set",
        )
    })?;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{}/search", base_url.trim_end_matches('/')))
        .query(&[
            ("q", request.query.as_str()),
            ("format", "json"),
            ("pageno", "1"),
        ])
        .send()
        .await
        .map_err(|e| api_error(StatusCode::BAD_GATEWAY, "proxy_error", &e.to_string()))?;

    if !resp.status().is_success() {
        let status = StatusCode::from_u16(resp.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let body = resp.text().await.unwrap_or_default();
        return Err(api_error(status, "proxy_error", &body));
    }

    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| api_error(StatusCode::BAD_GATEWAY, "proxy_error", &e.to_string()))?;

    let results: Vec<SearchResult> = body["results"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .take(request.max_results as usize)
        .map(|r| SearchResult {
            title: r["title"].as_str().unwrap_or("").to_string(),
            url: r["url"].as_str().unwrap_or("").to_string(),
            snippet: r["content"].as_str().unwrap_or("").to_string(),
            source: "searxng".to_string(),
        })
        .collect();

    let total = results.len() as u32;
    Ok(SearchResponse {
        results,
        query: request.query.clone(),
        total_results: total,
    })
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
        Some(key) if state.config.auth.api_keys.iter().any(|k| k.expose_secret() == key) => Ok(()),
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
