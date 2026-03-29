use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, error};

use super::types::{ChatRequest, ChatResponse, ResponseFormatType, StreamChunk, Usage};
use super::{Provider, ProviderError};

pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

impl AnthropicProvider {
    pub fn new(api_key: String, base_url: Option<String>) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url: base_url.unwrap_or_else(|| "https://api.anthropic.com".to_string()),
        }
    }
}

// ── Anthropic-specific request/response types ──

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AnthropicResponse {
    id: String,
    model: String,
    content: Vec<AnthropicContent>,
    usage: AnthropicUsage,
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AnthropicContent {
    #[serde(rename = "type")]
    content_type: String,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u64,
    output_tokens: u64,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(default)]
    delta: Option<AnthropicDelta>,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AnthropicDelta {
    #[serde(rename = "type")]
    delta_type: Option<String>,
    text: Option<String>,
    stop_reason: Option<String>,
}

// ── Format translation ──

fn to_anthropic_request(req: &ChatRequest, stream: bool) -> AnthropicRequest {
    let mut system_message = None;
    let mut messages = Vec::new();

    for msg in &req.messages {
        if msg.role == "system" {
            system_message = msg.content.clone();
        } else {
            messages.push(AnthropicMessage {
                role: msg.role.clone(),
                content: msg.content.clone().unwrap_or_default(),
            });
        }
    }

    // Handle response_format: for json_object or json_schema mode, append an
    // instruction to the system message so that Claude returns valid JSON.
    if let Some(ref rf) = req.response_format {
        match rf.format_type {
            ResponseFormatType::JsonObject => {
                let suffix = "Respond with valid JSON only.";
                system_message = Some(match system_message {
                    Some(existing) => format!("{}\n\n{}", existing, suffix),
                    None => suffix.to_string(),
                });
            }
            ResponseFormatType::JsonSchema => {
                let schema_instruction = if let Some(ref js) = rf.json_schema {
                    if let Some(ref schema) = js.schema {
                        format!(
                            "Respond with valid JSON only that conforms to this JSON schema: {}",
                            serde_json::to_string(schema).unwrap_or_default()
                        )
                    } else {
                        "Respond with valid JSON only.".to_string()
                    }
                } else {
                    "Respond with valid JSON only.".to_string()
                };
                system_message = Some(match system_message {
                    Some(existing) => format!("{}\n\n{}", existing, schema_instruction),
                    None => schema_instruction,
                });
            }
            ResponseFormatType::Text => {}
        }
    }

    AnthropicRequest {
        model: req.model.clone(),
        max_tokens: req.max_tokens.unwrap_or(4096),
        system: system_message,
        messages,
        temperature: req.temperature,
        top_p: req.top_p,
        stop_sequences: req.stop.clone(),
        stream,
    }
}

fn anthropic_to_chat_response(resp: AnthropicResponse) -> ChatResponse {
    let content = resp
        .content
        .iter()
        .filter_map(|c| c.text.clone())
        .collect::<Vec<_>>()
        .join("");

    ChatResponse::new(
        resp.model,
        content,
        Some(Usage {
            prompt_tokens: resp.usage.input_tokens,
            completion_tokens: resp.usage.output_tokens,
            total_tokens: resp.usage.input_tokens + resp.usage.output_tokens,
        }),
    )
}

fn map_stop_reason(reason: &str) -> &str {
    match reason {
        "end_turn" => "stop",
        "max_tokens" => "length",
        "stop_sequence" => "stop",
        other => other,
    }
}

// ── Provider implementation ──

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        debug!("Anthropic chat request for model: {}", request.model);

        let url = format!("{}/v1/messages", self.base_url);
        let anthropic_req = to_anthropic_request(request, false);

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&anthropic_req)
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        let status = response.status().as_u16();
        if status >= 400 {
            let body = response.text().await.unwrap_or_default();
            error!("Anthropic API error {}: {}", status, body);
            return Err(ProviderError::Api {
                status,
                message: body,
            });
        }

        let anthropic_resp: AnthropicResponse = response
            .json()
            .await
            .map_err(|e| ProviderError::Parse(e.to_string()))?;

        Ok(anthropic_to_chat_response(anthropic_resp))
    }

    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, ProviderError>>, ProviderError> {
        debug!("Anthropic streaming request for model: {}", request.model);

        let url = format!("{}/v1/messages", self.base_url);
        let anthropic_req = to_anthropic_request(request, true);

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&anthropic_req)
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
                    parse_anthropic_sse(&text, &model)
                }
                Err(e) => vec![Err(ProviderError::Network(e.to_string()))],
            })
            .flat_map(futures::stream::iter);

        Ok(Box::pin(stream))
    }
}

/// Parse Anthropic SSE events and translate to OpenAI-compatible StreamChunks
fn parse_anthropic_sse(text: &str, model: &str) -> Vec<Result<StreamChunk, ProviderError>> {
    let mut chunks = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data: ") {
            match serde_json::from_str::<AnthropicStreamEvent>(data) {
                Ok(event) => {
                    match event.event_type.as_str() {
                        "content_block_delta" => {
                            if let Some(delta) = &event.delta {
                                if let Some(text) = &delta.text {
                                    chunks.push(Ok(StreamChunk::new(
                                        model,
                                        Some(text.clone()),
                                        None,
                                    )));
                                }
                            }
                        }
                        "message_delta" => {
                            if let Some(delta) = &event.delta {
                                if let Some(reason) = &delta.stop_reason {
                                    chunks.push(Ok(StreamChunk::new(
                                        model,
                                        None,
                                        Some(map_stop_reason(reason).to_string()),
                                    )));
                                }
                            }
                        }
                        _ => {} // Ignore other event types
                    }
                }
                Err(_) => {
                    debug!("Skipping unparseable Anthropic SSE data: {}", data);
                }
            }
        }
    }

    chunks
}
