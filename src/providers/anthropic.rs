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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::types::{JsonSchemaFormat, Message, ResponseFormat, ResponseFormatType};
    use serde_json::json;

    // ── Helper functions ──

    fn create_chat_request(
        model: &str,
        messages: Vec<Message>,
        max_tokens: Option<u64>,
        response_format: Option<ResponseFormat>,
    ) -> ChatRequest {
        ChatRequest {
            model: model.to_string(),
            messages,
            temperature: None,
            max_tokens,
            top_p: None,
            stream: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            tools: None,
            tool_choice: None,
            response_format,
        }
    }

    fn create_message(role: &str, content: Option<String>) -> Message {
        Message {
            role: role.to_string(),
            content,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    // ── Tests for to_anthropic_request ──

    #[test]
    fn test_system_message_extraction() {
        let messages = vec![
            create_message("system", Some("You are a helpful assistant".to_string())),
            create_message("user", Some("Hello".to_string())),
        ];
        let req = create_chat_request("claude-3-sonnet", messages, None, None);

        let result = to_anthropic_request(&req, false);

        assert_eq!(
            result.system,
            Some("You are a helpful assistant".to_string())
        );
        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0].role, "user");
        assert_eq!(result.messages[0].content, "Hello");
    }

    #[test]
    fn test_non_system_messages_passthrough() {
        let messages = vec![
            create_message("user", Some("What is 2+2?".to_string())),
            create_message("assistant", Some("2+2 equals 4.".to_string())),
        ];
        let req = create_chat_request("claude-3-sonnet", messages, None, None);

        let result = to_anthropic_request(&req, false);

        assert_eq!(result.system, None);
        assert_eq!(result.messages.len(), 2);
        assert_eq!(result.messages[0].role, "user");
        assert_eq!(result.messages[0].content, "What is 2+2?");
        assert_eq!(result.messages[1].role, "assistant");
        assert_eq!(result.messages[1].content, "2+2 equals 4.");
    }

    #[test]
    fn test_max_tokens_default() {
        let messages = vec![create_message("user", Some("Hi".to_string()))];
        let req = create_chat_request("claude-3-sonnet", messages, None, None);

        let result = to_anthropic_request(&req, false);

        assert_eq!(result.max_tokens, 4096);
    }

    #[test]
    fn test_max_tokens_passthrough() {
        let messages = vec![create_message("user", Some("Hi".to_string()))];
        let req = create_chat_request("claude-3-sonnet", messages, Some(2048), None);

        let result = to_anthropic_request(&req, false);

        assert_eq!(result.max_tokens, 2048);
    }

    #[test]
    fn test_response_format_json_object() {
        let messages = vec![create_message("user", Some("Generate JSON".to_string()))];
        let response_format = ResponseFormat {
            format_type: ResponseFormatType::JsonObject,
            json_schema: None,
        };
        let req = create_chat_request("claude-3-sonnet", messages, None, Some(response_format));

        let result = to_anthropic_request(&req, false);

        assert!(result.system.is_some());
        let system = result.system.unwrap();
        assert!(system.contains("Respond with valid JSON only."));
    }

    #[test]
    fn test_response_format_json_object_with_existing_system() {
        let messages = vec![create_message("user", Some("Generate JSON".to_string()))];
        let response_format = ResponseFormat {
            format_type: ResponseFormatType::JsonObject,
            json_schema: None,
        };
        let mut req = create_chat_request("claude-3-sonnet", messages, None, Some(response_format));
        // Manually set system message
        req.messages
            .insert(0, create_message("system", Some("Be concise".to_string())));

        let result = to_anthropic_request(&req, false);

        assert!(result.system.is_some());
        let system = result.system.unwrap();
        assert!(system.contains("Be concise"));
        assert!(system.contains("Respond with valid JSON only."));
        assert!(system.contains("\n\n"));
    }

    #[test]
    fn test_response_format_json_schema_with_schema() {
        let messages = vec![create_message(
            "user",
            Some("Generate structured data".to_string()),
        )];
        let schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            }
        });
        let response_format = ResponseFormat {
            format_type: ResponseFormatType::JsonSchema,
            json_schema: Some(JsonSchemaFormat {
                name: "TestSchema".to_string(),
                description: None,
                schema: Some(schema),
                strict: None,
            }),
        };
        let req = create_chat_request("claude-3-sonnet", messages, None, Some(response_format));

        let result = to_anthropic_request(&req, false);

        assert!(result.system.is_some());
        let system = result.system.unwrap();
        assert!(system.contains("Respond with valid JSON only that conforms to this JSON schema:"));
        assert!(system.contains("\"type\":\"object\""));
    }

    #[test]
    fn test_response_format_json_schema_without_schema() {
        let messages = vec![create_message("user", Some("Generate JSON".to_string()))];
        let response_format = ResponseFormat {
            format_type: ResponseFormatType::JsonSchema,
            json_schema: Some(JsonSchemaFormat {
                name: "TestSchema".to_string(),
                description: None,
                schema: None,
                strict: None,
            }),
        };
        let req = create_chat_request("claude-3-sonnet", messages, None, Some(response_format));

        let result = to_anthropic_request(&req, false);

        assert!(result.system.is_some());
        let system = result.system.unwrap();
        assert_eq!(system, "Respond with valid JSON only.");
    }

    #[test]
    fn test_response_format_text() {
        let messages = vec![create_message("user", Some("Just talk".to_string()))];
        let response_format = ResponseFormat {
            format_type: ResponseFormatType::Text,
            json_schema: None,
        };
        let req = create_chat_request("claude-3-sonnet", messages, None, Some(response_format));

        let result = to_anthropic_request(&req, false);

        assert_eq!(result.system, None);
    }

    #[test]
    fn test_stream_flag_passed_through() {
        let messages = vec![create_message("user", Some("Hi".to_string()))];
        let req = create_chat_request("claude-3-sonnet", messages, None, None);

        let result_stream = to_anthropic_request(&req, true);
        let result_no_stream = to_anthropic_request(&req, false);

        assert!(result_stream.stream);
        assert!(!result_no_stream.stream);
    }

    // ── Tests for anthropic_to_chat_response ──

    #[test]
    fn test_anthropic_response_to_chat_response() {
        let resp = AnthropicResponse {
            id: "msg_123".to_string(),
            model: "claude-3-sonnet".to_string(),
            content: vec![AnthropicContent {
                content_type: "text".to_string(),
                text: Some("Hello, world!".to_string()),
            }],
            usage: AnthropicUsage {
                input_tokens: 10,
                output_tokens: 5,
            },
            stop_reason: None,
        };

        let result = anthropic_to_chat_response(resp);

        assert_eq!(result.model, "claude-3-sonnet");
        assert_eq!(result.choices.len(), 1);
        assert_eq!(
            result.choices[0].message.content,
            Some("Hello, world!".to_string())
        );
        assert!(result.usage.is_some());

        let usage = result.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn test_anthropic_response_multiple_content_blocks() {
        let resp = AnthropicResponse {
            id: "msg_456".to_string(),
            model: "claude-3-opus".to_string(),
            content: vec![
                AnthropicContent {
                    content_type: "text".to_string(),
                    text: Some("Part 1".to_string()),
                },
                AnthropicContent {
                    content_type: "text".to_string(),
                    text: Some(" Part 2".to_string()),
                },
                AnthropicContent {
                    content_type: "text".to_string(),
                    text: Some(" Part 3".to_string()),
                },
            ],
            usage: AnthropicUsage {
                input_tokens: 20,
                output_tokens: 10,
            },
            stop_reason: None,
        };

        let result = anthropic_to_chat_response(resp);

        assert_eq!(
            result.choices[0].message.content,
            Some("Part 1 Part 2 Part 3".to_string())
        );
    }

    #[test]
    fn test_anthropic_response_skips_non_text_content() {
        let resp = AnthropicResponse {
            id: "msg_789".to_string(),
            model: "claude-3-sonnet".to_string(),
            content: vec![
                AnthropicContent {
                    content_type: "text".to_string(),
                    text: Some("Text content".to_string()),
                },
                AnthropicContent {
                    content_type: "image".to_string(),
                    text: None,
                },
                AnthropicContent {
                    content_type: "text".to_string(),
                    text: None,
                },
            ],
            usage: AnthropicUsage {
                input_tokens: 15,
                output_tokens: 8,
            },
            stop_reason: None,
        };

        let result = anthropic_to_chat_response(resp);

        // Should only include the text content, not None or non-text types
        assert_eq!(
            result.choices[0].message.content,
            Some("Text content".to_string())
        );
    }

    #[test]
    fn test_anthropic_response_empty_content() {
        let resp = AnthropicResponse {
            id: "msg_empty".to_string(),
            model: "claude-3-sonnet".to_string(),
            content: vec![],
            usage: AnthropicUsage {
                input_tokens: 5,
                output_tokens: 0,
            },
            stop_reason: None,
        };

        let result = anthropic_to_chat_response(resp);

        assert_eq!(result.choices[0].message.content, Some("".to_string()));
    }

    // ── Tests for map_stop_reason ──

    #[test]
    fn test_map_stop_reason_end_turn() {
        assert_eq!(map_stop_reason("end_turn"), "stop");
    }

    #[test]
    fn test_map_stop_reason_max_tokens() {
        assert_eq!(map_stop_reason("max_tokens"), "length");
    }

    #[test]
    fn test_map_stop_reason_stop_sequence() {
        assert_eq!(map_stop_reason("stop_sequence"), "stop");
    }

    #[test]
    fn test_map_stop_reason_unknown() {
        assert_eq!(map_stop_reason("unknown_reason"), "unknown_reason");
    }

    #[test]
    fn test_map_stop_reason_passthrough() {
        assert_eq!(map_stop_reason("custom_stop"), "custom_stop");
    }
}
