use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, error};

use super::types::{ChatRequest, ChatResponse, StreamChunk, Usage};
use super::{Provider, ProviderError};

pub struct GeminiProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

impl GeminiProvider {
    pub fn new(api_key: String, base_url: Option<String>) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url: base_url
                .unwrap_or_else(|| "https://generativelanguage.googleapis.com".to_string()),
        }
    }
}

// ── Gemini-specific request/response types ──

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
struct GeminiCandidate {
    content: Option<GeminiContent>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsageMetadata {
    #[serde(default)]
    prompt_token_count: u64,
    #[serde(default)]
    candidates_token_count: u64,
    #[serde(default)]
    total_token_count: u64,
}

// ── Format translation ──

fn to_gemini_request(req: &ChatRequest) -> GeminiRequest {
    let mut system_instruction = None;
    let mut contents = Vec::new();

    for msg in &req.messages {
        let text = msg.content.clone().unwrap_or_default();

        if msg.role == "system" {
            system_instruction = Some(GeminiContent {
                role: None,
                parts: vec![GeminiPart { text: Some(text) }],
            });
        } else {
            // Gemini uses "user" and "model" instead of "user" and "assistant"
            let role = match msg.role.as_str() {
                "assistant" => "model".to_string(),
                other => other.to_string(),
            };
            contents.push(GeminiContent {
                role: Some(role),
                parts: vec![GeminiPart { text: Some(text) }],
            });
        }
    }

    let generation_config =
        if req.temperature.is_some() || req.top_p.is_some() || req.max_tokens.is_some() {
            Some(GenerationConfig {
                temperature: req.temperature,
                top_p: req.top_p,
                max_output_tokens: req.max_tokens,
                stop_sequences: req.stop.clone(),
            })
        } else {
            None
        };

    GeminiRequest {
        contents,
        system_instruction,
        generation_config,
    }
}

fn gemini_to_chat_response(resp: GeminiResponse, model: &str) -> ChatResponse {
    let content = resp
        .candidates
        .as_ref()
        .and_then(|c| c.first())
        .and_then(|c| c.content.as_ref())
        .and_then(|c| c.parts.first())
        .and_then(|p| p.text.clone())
        .unwrap_or_default();

    let usage = resp.usage_metadata.map(|u| Usage {
        prompt_tokens: u.prompt_token_count,
        completion_tokens: u.candidates_token_count,
        total_tokens: u.total_token_count,
    });

    ChatResponse::new(model.to_string(), content, usage)
}

fn map_finish_reason(reason: &str) -> &str {
    match reason {
        "STOP" => "stop",
        "MAX_TOKENS" => "length",
        "SAFETY" => "content_filter",
        other => other,
    }
}

// ── Provider implementation ──

#[async_trait]
impl Provider for GeminiProvider {
    fn name(&self) -> &str {
        "gemini"
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        debug!("Gemini chat request for model: {}", request.model);

        let url = format!(
            "{}/v1beta/models/{}:generateContent?key={}",
            self.base_url, request.model, self.api_key
        );
        let gemini_req = to_gemini_request(request);

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&gemini_req)
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        let status = response.status().as_u16();
        if status >= 400 {
            let body = response.text().await.unwrap_or_default();
            error!("Gemini API error {}: {}", status, body);
            return Err(ProviderError::Api {
                status,
                message: body,
            });
        }

        let gemini_resp: GeminiResponse = response
            .json()
            .await
            .map_err(|e| ProviderError::Parse(e.to_string()))?;

        Ok(gemini_to_chat_response(gemini_resp, &request.model))
    }

    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, ProviderError>>, ProviderError> {
        debug!("Gemini streaming request for model: {}", request.model);

        let url = format!(
            "{}/v1beta/models/{}:streamGenerateContent?alt=sse&key={}",
            self.base_url, request.model, self.api_key
        );
        let gemini_req = to_gemini_request(request);

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&gemini_req)
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        let status = response.status().as_u16();
        if status >= 400 {
            let body = response.text().await.unwrap_or_default();
            return Err(ProviderError::Api {
                status,
                message: body,
            });
        }

        let model = request.model.clone();
        let stream = response
            .bytes_stream()
            .map(move |result| match result {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    parse_gemini_sse(&text, &model)
                }
                Err(e) => vec![Err(ProviderError::Network(e.to_string()))],
            })
            .flat_map(futures::stream::iter);

        Ok(Box::pin(stream))
    }
}

/// Parse Gemini SSE events and translate to OpenAI-compatible StreamChunks
fn parse_gemini_sse(text: &str, model: &str) -> Vec<Result<StreamChunk, ProviderError>> {
    let mut chunks = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data: ") {
            match serde_json::from_str::<GeminiResponse>(data) {
                Ok(resp) => {
                    if let Some(candidates) = &resp.candidates {
                        if let Some(candidate) = candidates.first() {
                            // Extract text content
                            let content = candidate
                                .content
                                .as_ref()
                                .and_then(|c| c.parts.first())
                                .and_then(|p| p.text.clone());

                            // Map finish reason
                            let finish_reason = candidate
                                .finish_reason
                                .as_ref()
                                .map(|r| map_finish_reason(r).to_string());

                            if content.is_some() || finish_reason.is_some() {
                                chunks.push(Ok(StreamChunk::new(model, content, finish_reason)));
                            }
                        }
                    }
                }
                Err(_) => {
                    debug!("Skipping unparseable Gemini SSE data: {}", data);
                }
            }
        }
    }

    chunks
}
