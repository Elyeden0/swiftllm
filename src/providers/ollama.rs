use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, error};

use super::types::{ChatRequest, ChatResponse, ResponseFormatType, StreamChunk, Usage};
use super::{Provider, ProviderError};

pub struct OllamaProvider {
    client: Client,
    base_url: String,
}

impl OllamaProvider {
    pub fn new(base_url: Option<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.unwrap_or_else(|| "http://localhost:11434".to_string()),
        }
    }
}

// ── Ollama-specific types ──

#[derive(Debug, Serialize)]
struct OllamaRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
    /// When set to "json", Ollama returns structured JSON output.
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    model: String,
    message: OllamaMessage,
    done: bool,
    #[serde(default)]
    prompt_eval_count: Option<u64>,
    #[serde(default)]
    eval_count: Option<u64>,
}

// ── Format translation ──

fn to_ollama_request(req: &ChatRequest, stream: bool) -> OllamaRequest {
    let messages = req
        .messages
        .iter()
        .map(|m| OllamaMessage {
            role: m.role.clone(),
            content: m.content.clone().unwrap_or_default(),
        })
        .collect();

    let options = if req.temperature.is_some()
        || req.top_p.is_some()
        || req.max_tokens.is_some()
        || req.stop.is_some()
    {
        Some(OllamaOptions {
            temperature: req.temperature,
            top_p: req.top_p,
            num_predict: req.max_tokens,
            stop: req.stop.clone(),
        })
    } else {
        None
    };

    // Translate response_format to Ollama's format field
    let format = req
        .response_format
        .as_ref()
        .and_then(|rf| match rf.format_type {
            ResponseFormatType::JsonObject => Some(serde_json::Value::String("json".to_string())),
            ResponseFormatType::JsonSchema => {
                // Ollama supports passing a JSON schema object directly as the format value
                rf.json_schema
                    .as_ref()
                    .and_then(|js| js.schema.clone())
                    .or_else(|| Some(serde_json::Value::String("json".to_string())))
            }
            ResponseFormatType::Text => None,
        });

    OllamaRequest {
        model: req.model.clone(),
        messages,
        stream,
        options,
        format,
    }
}

// ── Provider implementation ──

#[async_trait]
impl Provider for OllamaProvider {
    fn name(&self) -> &str {
        "ollama"
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        debug!("Ollama chat request for model: {}", request.model);

        let url = format!("{}/api/chat", self.base_url);
        let ollama_req = to_ollama_request(request, false);

        let response = self
            .client
            .post(&url)
            .json(&ollama_req)
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        let status = response.status().as_u16();
        if status >= 400 {
            let body = response.text().await.unwrap_or_default();
            error!("Ollama API error {}: {}", status, body);
            return Err(ProviderError::Api {
                status,
                message: body,
            });
        }

        let ollama_resp: OllamaResponse = response
            .json()
            .await
            .map_err(|e| ProviderError::Parse(e.to_string()))?;

        let usage = match (ollama_resp.prompt_eval_count, ollama_resp.eval_count) {
            (Some(input), Some(output)) => Some(Usage {
                prompt_tokens: input,
                completion_tokens: output,
                total_tokens: input + output,
            }),
            _ => None,
        };

        Ok(ChatResponse::new(
            ollama_resp.model,
            ollama_resp.message.content,
            usage,
        ))
    }

    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, ProviderError>>, ProviderError> {
        debug!("Ollama streaming request for model: {}", request.model);

        let url = format!("{}/api/chat", self.base_url);
        let ollama_req = to_ollama_request(request, true);

        let response = self
            .client
            .post(&url)
            .json(&ollama_req)
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
                    parse_ollama_stream(&text, &model)
                }
                Err(e) => vec![Err(ProviderError::Network(e.to_string()))],
            })
            .flat_map(futures::stream::iter);

        Ok(Box::pin(stream))
    }
}

/// Parse Ollama's NDJSON stream into OpenAI-compatible StreamChunks
fn parse_ollama_stream(text: &str, model: &str) -> Vec<Result<StreamChunk, ProviderError>> {
    let mut chunks = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<OllamaResponse>(line) {
            Ok(resp) => {
                if resp.done {
                    chunks.push(Ok(StreamChunk::new(model, None, Some("stop".to_string()))));
                } else {
                    chunks.push(Ok(StreamChunk::new(
                        model,
                        Some(resp.message.content),
                        None,
                    )));
                }
            }
            Err(e) => {
                debug!("Failed to parse Ollama stream line: {} — {}", e, line);
            }
        }
    }

    chunks
}
