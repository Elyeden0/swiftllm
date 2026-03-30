use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, error};

use super::types::{
    ChatRequest, ChatResponse, EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage,
    ResponseFormatType, StreamChunk, Usage,
};
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
    #[serde(skip_serializing_if = "Option::is_none")]
    response_mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_schema: Option<serde_json::Value>,
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

    // Translate response_format to Gemini's responseMimeType / responseSchema
    let (response_mime_type, response_schema) = match &req.response_format {
        Some(rf) => match rf.format_type {
            ResponseFormatType::JsonObject => (Some("application/json".to_string()), None),
            ResponseFormatType::JsonSchema => {
                let schema = rf.json_schema.as_ref().and_then(|js| js.schema.clone());
                (Some("application/json".to_string()), schema)
            }
            ResponseFormatType::Text => (None, None),
        },
        None => (None, None),
    };

    let needs_config = req.temperature.is_some()
        || req.top_p.is_some()
        || req.max_tokens.is_some()
        || response_mime_type.is_some();

    let generation_config = if needs_config {
        Some(GenerationConfig {
            temperature: req.temperature,
            top_p: req.top_p,
            max_output_tokens: req.max_tokens,
            stop_sequences: req.stop.clone(),
            response_mime_type,
            response_schema,
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

    async fn embeddings(
        &self,
        request: &EmbeddingRequest,
    ) -> Result<EmbeddingResponse, ProviderError> {
        debug!("Gemini embeddings request for model: {}", request.model);

        let inputs = request.input.to_vec();
        let mut all_data = Vec::new();

        for (idx, text) in inputs.iter().enumerate() {
            let url = format!(
                "{}/v1beta/models/{}:embedContent?key={}",
                self.base_url, request.model, self.api_key
            );

            let gemini_embed_req = GeminiEmbedRequest {
                content: GeminiContent {
                    role: None,
                    parts: vec![GeminiPart {
                        text: Some(text.clone()),
                    }],
                },
            };

            let response = self
                .client
                .post(&url)
                .header("Content-Type", "application/json")
                .json(&gemini_embed_req)
                .send()
                .await
                .map_err(|e| ProviderError::Network(e.to_string()))?;

            let status = response.status().as_u16();
            if status >= 400 {
                let body = response.text().await.unwrap_or_default();
                error!("Gemini embeddings API error {}: {}", status, body);
                return Err(ProviderError::Api {
                    status,
                    message: body,
                });
            }

            let gemini_resp: GeminiEmbedResponse = response
                .json()
                .await
                .map_err(|e| ProviderError::Parse(e.to_string()))?;

            all_data.push(EmbeddingData {
                object: "embedding".to_string(),
                embedding: gemini_resp.embedding.values,
                index: idx,
            });
        }

        Ok(EmbeddingResponse::new(
            request.model.clone(),
            all_data,
            EmbeddingUsage {
                prompt_tokens: 0,
                total_tokens: 0,
            },
        ))
    }
}

// ── Gemini embedding types ──

#[derive(Debug, Serialize)]
struct GeminiEmbedRequest {
    content: GeminiContent,
}

#[derive(Debug, Deserialize)]
struct GeminiEmbedResponse {
    embedding: GeminiEmbedding,
}

#[derive(Debug, Deserialize)]
struct GeminiEmbedding {
    values: Vec<f64>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::types::{JsonSchemaFormat, Message, ResponseFormat, ResponseFormatType};

    fn create_basic_chat_request() -> ChatRequest {
        ChatRequest {
            model: "gemini-pro".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: Some("Hello, world!".to_string()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            temperature: None,
            max_tokens: None,
            top_p: None,
            stream: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        }
    }

    // ── Role mapping tests ──

    #[test]
    fn test_assistant_role_maps_to_model() {
        let request = ChatRequest {
            messages: vec![Message {
                role: "assistant".to_string(),
                content: Some("Hello".to_string()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            ..create_basic_chat_request()
        };

        let gemini_req = to_gemini_request(&request);
        assert_eq!(gemini_req.contents.len(), 1);
        assert_eq!(gemini_req.contents[0].role, Some("model".to_string()));
    }

    #[test]
    fn test_user_role_stays_user() {
        let request = ChatRequest {
            messages: vec![Message {
                role: "user".to_string(),
                content: Some("Hello".to_string()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            ..create_basic_chat_request()
        };

        let gemini_req = to_gemini_request(&request);
        assert_eq!(gemini_req.contents.len(), 1);
        assert_eq!(gemini_req.contents[0].role, Some("user".to_string()));
    }

    #[test]
    fn test_system_message_to_system_instruction() {
        let request = ChatRequest {
            messages: vec![
                Message {
                    role: "system".to_string(),
                    content: Some("You are a helpful assistant".to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
                Message {
                    role: "user".to_string(),
                    content: Some("Hello".to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
            ],
            ..create_basic_chat_request()
        };

        let gemini_req = to_gemini_request(&request);
        assert!(gemini_req.system_instruction.is_some());
        let system = gemini_req.system_instruction.unwrap();
        assert_eq!(system.role, None);
        assert_eq!(system.parts.len(), 1);
        assert_eq!(
            system.parts[0].text,
            Some("You are a helpful assistant".to_string())
        );
        assert_eq!(gemini_req.contents.len(), 1); // Only user message in contents
    }

    #[test]
    fn test_system_message_excluded_from_contents() {
        let request = ChatRequest {
            messages: vec![
                Message {
                    role: "system".to_string(),
                    content: Some("System prompt".to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
                Message {
                    role: "user".to_string(),
                    content: Some("User message".to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
                Message {
                    role: "assistant".to_string(),
                    content: Some("Assistant message".to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
            ],
            ..create_basic_chat_request()
        };

        let gemini_req = to_gemini_request(&request);
        assert_eq!(gemini_req.contents.len(), 2);
        assert_eq!(gemini_req.contents[0].role, Some("user".to_string()));
        assert_eq!(gemini_req.contents[1].role, Some("model".to_string()));
    }

    // ── GenerationConfig tests ──

    #[test]
    fn test_temperature_maps_to_generation_config() {
        let request = ChatRequest {
            temperature: Some(0.7),
            ..create_basic_chat_request()
        };

        let gemini_req = to_gemini_request(&request);
        assert!(gemini_req.generation_config.is_some());
        let config = gemini_req.generation_config.unwrap();
        assert_eq!(config.temperature, Some(0.7));
    }

    #[test]
    fn test_top_p_maps_to_generation_config() {
        let request = ChatRequest {
            top_p: Some(0.95),
            ..create_basic_chat_request()
        };

        let gemini_req = to_gemini_request(&request);
        assert!(gemini_req.generation_config.is_some());
        let config = gemini_req.generation_config.unwrap();
        assert_eq!(config.top_p, Some(0.95));
    }

    #[test]
    fn test_max_tokens_maps_to_max_output_tokens() {
        let request = ChatRequest {
            max_tokens: Some(1024),
            ..create_basic_chat_request()
        };

        let gemini_req = to_gemini_request(&request);
        assert!(gemini_req.generation_config.is_some());
        let config = gemini_req.generation_config.unwrap();
        assert_eq!(config.max_output_tokens, Some(1024));
    }

    #[test]
    fn test_stop_sequences_map_to_generation_config() {
        // stop_sequences alone don't trigger generation_config (the code checks
        // temperature, top_p, max_tokens, and response_mime_type). We need to
        // also set one of those to get a generation_config.
        let request = ChatRequest {
            stop: Some(vec!["END".to_string(), "STOP".to_string()]),
            temperature: Some(0.5),
            ..create_basic_chat_request()
        };

        let gemini_req = to_gemini_request(&request);
        assert!(gemini_req.generation_config.is_some());
        let config = gemini_req.generation_config.unwrap();
        assert_eq!(
            config.stop_sequences,
            Some(vec!["END".to_string(), "STOP".to_string()])
        );
    }

    #[test]
    fn test_multiple_generation_config_options() {
        let request = ChatRequest {
            temperature: Some(0.5),
            top_p: Some(0.9),
            max_tokens: Some(2048),
            stop: Some(vec!["END".to_string()]),
            ..create_basic_chat_request()
        };

        let gemini_req = to_gemini_request(&request);
        assert!(gemini_req.generation_config.is_some());
        let config = gemini_req.generation_config.unwrap();
        assert_eq!(config.temperature, Some(0.5));
        assert_eq!(config.top_p, Some(0.9));
        assert_eq!(config.max_output_tokens, Some(2048));
        assert_eq!(config.stop_sequences, Some(vec!["END".to_string()]));
    }

    #[test]
    fn test_no_generation_config_when_all_none() {
        let request = create_basic_chat_request();
        let gemini_req = to_gemini_request(&request);
        assert!(gemini_req.generation_config.is_none());
    }

    // ── Response format tests ──

    #[test]
    fn test_response_format_json_object() {
        let request = ChatRequest {
            response_format: Some(ResponseFormat {
                format_type: ResponseFormatType::JsonObject,
                json_schema: None,
            }),
            ..create_basic_chat_request()
        };

        let gemini_req = to_gemini_request(&request);
        assert!(gemini_req.generation_config.is_some());
        let config = gemini_req.generation_config.unwrap();
        assert_eq!(
            config.response_mime_type,
            Some("application/json".to_string())
        );
        assert!(config.response_schema.is_none());
    }

    #[test]
    fn test_response_format_json_schema() {
        let schema_json = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer" }
            }
        });

        let request = ChatRequest {
            response_format: Some(ResponseFormat {
                format_type: ResponseFormatType::JsonSchema,
                json_schema: Some(JsonSchemaFormat {
                    name: "Person".to_string(),
                    description: Some("A person schema".to_string()),
                    schema: Some(schema_json.clone()),
                    strict: Some(true),
                }),
            }),
            ..create_basic_chat_request()
        };

        let gemini_req = to_gemini_request(&request);
        assert!(gemini_req.generation_config.is_some());
        let config = gemini_req.generation_config.unwrap();
        assert_eq!(
            config.response_mime_type,
            Some("application/json".to_string())
        );
        assert_eq!(config.response_schema, Some(schema_json));
    }

    #[test]
    fn test_response_format_text() {
        let request = ChatRequest {
            response_format: Some(ResponseFormat {
                format_type: ResponseFormatType::Text,
                json_schema: None,
            }),
            ..create_basic_chat_request()
        };

        let gemini_req = to_gemini_request(&request);
        // Text format should not create generation_config if no other options
        if let Some(config) = gemini_req.generation_config {
            assert!(config.response_mime_type.is_none());
            assert!(config.response_schema.is_none());
        }
    }

    #[test]
    fn test_response_format_json_schema_with_no_schema() {
        let request = ChatRequest {
            response_format: Some(ResponseFormat {
                format_type: ResponseFormatType::JsonSchema,
                json_schema: None,
            }),
            ..create_basic_chat_request()
        };

        let gemini_req = to_gemini_request(&request);
        assert!(gemini_req.generation_config.is_some());
        let config = gemini_req.generation_config.unwrap();
        assert_eq!(
            config.response_mime_type,
            Some("application/json".to_string())
        );
        assert!(config.response_schema.is_none());
    }

    // ── GeminiResponse to ChatResponse tests ──

    #[test]
    fn test_gemini_response_to_chat_response_text_extraction() {
        let gemini_resp = GeminiResponse {
            candidates: Some(vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: Some("model".to_string()),
                    parts: vec![GeminiPart {
                        text: Some("This is the response".to_string()),
                    }],
                }),
                finish_reason: Some("STOP".to_string()),
            }]),
            usage_metadata: None,
        };

        let chat_resp = gemini_to_chat_response(gemini_resp, "gemini-pro");
        assert_eq!(chat_resp.model, "gemini-pro");
        assert_eq!(chat_resp.choices.len(), 1);
        assert_eq!(
            chat_resp.choices[0].message.content,
            Some("This is the response".to_string())
        );
    }

    #[test]
    fn test_gemini_response_usage_mapping() {
        let gemini_resp = GeminiResponse {
            candidates: Some(vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: Some("model".to_string()),
                    parts: vec![GeminiPart {
                        text: Some("Response".to_string()),
                    }],
                }),
                finish_reason: None,
            }]),
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: 10,
                candidates_token_count: 20,
                total_token_count: 30,
            }),
        };

        let chat_resp = gemini_to_chat_response(gemini_resp, "gemini-pro");
        assert!(chat_resp.usage.is_some());
        let usage = chat_resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
    }

    #[test]
    fn test_gemini_response_missing_candidates() {
        let gemini_resp = GeminiResponse {
            candidates: None,
            usage_metadata: None,
        };

        let chat_resp = gemini_to_chat_response(gemini_resp, "gemini-pro");
        assert_eq!(chat_resp.choices[0].message.content, Some(String::new()));
    }

    #[test]
    fn test_gemini_response_empty_candidates() {
        let gemini_resp = GeminiResponse {
            candidates: Some(vec![]),
            usage_metadata: None,
        };

        let chat_resp = gemini_to_chat_response(gemini_resp, "gemini-pro");
        assert_eq!(chat_resp.choices[0].message.content, Some(String::new()));
    }

    #[test]
    fn test_gemini_response_missing_text_in_parts() {
        let gemini_resp = GeminiResponse {
            candidates: Some(vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: Some("model".to_string()),
                    parts: vec![GeminiPart { text: None }],
                }),
                finish_reason: None,
            }]),
            usage_metadata: None,
        };

        let chat_resp = gemini_to_chat_response(gemini_resp, "gemini-pro");
        assert_eq!(chat_resp.choices[0].message.content, Some(String::new()));
    }

    // ── map_finish_reason tests ──

    #[test]
    fn test_map_finish_reason_stop() {
        assert_eq!(map_finish_reason("STOP"), "stop");
    }

    #[test]
    fn test_map_finish_reason_max_tokens() {
        assert_eq!(map_finish_reason("MAX_TOKENS"), "length");
    }

    #[test]
    fn test_map_finish_reason_safety() {
        assert_eq!(map_finish_reason("SAFETY"), "content_filter");
    }

    #[test]
    fn test_map_finish_reason_unknown() {
        assert_eq!(map_finish_reason("UNKNOWN"), "UNKNOWN");
    }

    #[test]
    fn test_map_finish_reason_case_sensitive() {
        assert_eq!(map_finish_reason("stop"), "stop"); // Should not match "STOP"
        assert_eq!(map_finish_reason("Stop"), "Stop"); // Should not match "STOP"
    }
}
