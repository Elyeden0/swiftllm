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

    async fn embeddings(
        &self,
        request: &EmbeddingRequest,
    ) -> Result<EmbeddingResponse, ProviderError> {
        debug!("Ollama embeddings request for model: {}", request.model);

        let url = format!("{}/api/embed", self.base_url);
        let inputs = request.input.to_vec();

        let ollama_embed_req = OllamaEmbedRequest {
            model: request.model.clone(),
            input: inputs,
        };

        let response = self
            .client
            .post(&url)
            .json(&ollama_embed_req)
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        let status = response.status().as_u16();
        if status >= 400 {
            let body = response.text().await.unwrap_or_default();
            error!("Ollama embeddings API error {}: {}", status, body);
            return Err(ProviderError::Api {
                status,
                message: body,
            });
        }

        let ollama_resp: OllamaEmbedResponse = response
            .json()
            .await
            .map_err(|e| ProviderError::Parse(e.to_string()))?;

        let data: Vec<EmbeddingData> = ollama_resp
            .embeddings
            .into_iter()
            .enumerate()
            .map(|(idx, embedding)| EmbeddingData {
                object: "embedding".to_string(),
                embedding,
                index: idx,
            })
            .collect();

        Ok(EmbeddingResponse::new(
            request.model.clone(),
            data,
            EmbeddingUsage {
                prompt_tokens: 0,
                total_tokens: 0,
            },
        ))
    }
}

// ── Ollama embedding types ──

#[derive(Debug, Serialize)]
struct OllamaEmbedRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f64>>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::types::{JsonSchemaFormat, Message, ResponseFormat, ResponseFormatType};

    fn create_test_request(model: &str) -> ChatRequest {
        ChatRequest {
            model: model.to_string(),
            messages: vec![],
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

    #[test]
    fn test_message_format_basic() {
        let mut req = create_test_request("llama2");
        req.messages = vec![
            Message {
                role: "user".to_string(),
                content: Some("Hello".to_string()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            Message {
                role: "assistant".to_string(),
                content: Some("Hi there!".to_string()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ];

        let ollama_req = to_ollama_request(&req, false);

        assert_eq!(ollama_req.messages.len(), 2);
        assert_eq!(ollama_req.messages[0].role, "user");
        assert_eq!(ollama_req.messages[0].content, "Hello");
        assert_eq!(ollama_req.messages[1].role, "assistant");
        assert_eq!(ollama_req.messages[1].content, "Hi there!");
    }

    #[test]
    fn test_message_format_system_passthrough() {
        let mut req = create_test_request("llama2");
        req.messages = vec![
            Message {
                role: "system".to_string(),
                content: Some("You are a helpful assistant.".to_string()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            Message {
                role: "user".to_string(),
                content: Some("Question?".to_string()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ];

        let ollama_req = to_ollama_request(&req, false);

        assert_eq!(ollama_req.messages[0].role, "system");
        assert_eq!(
            ollama_req.messages[0].content,
            "You are a helpful assistant."
        );
    }

    #[test]
    fn test_message_empty_content_defaults() {
        let mut req = create_test_request("llama2");
        req.messages = vec![Message {
            role: "user".to_string(),
            content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];

        let ollama_req = to_ollama_request(&req, false);

        assert_eq!(ollama_req.messages[0].content, "");
    }

    #[test]
    fn test_options_mapping_temperature() {
        let mut req = create_test_request("llama2");
        req.temperature = Some(0.7);

        let ollama_req = to_ollama_request(&req, false);

        assert!(ollama_req.options.is_some());
        let opts = ollama_req.options.unwrap();
        assert_eq!(opts.temperature, Some(0.7));
        assert_eq!(opts.top_p, None);
        assert_eq!(opts.num_predict, None);
        assert_eq!(opts.stop, None);
    }

    #[test]
    fn test_options_mapping_top_p() {
        let mut req = create_test_request("llama2");
        req.top_p = Some(0.9);

        let ollama_req = to_ollama_request(&req, false);

        assert!(ollama_req.options.is_some());
        let opts = ollama_req.options.unwrap();
        assert_eq!(opts.temperature, None);
        assert_eq!(opts.top_p, Some(0.9));
    }

    #[test]
    fn test_options_mapping_max_tokens_to_num_predict() {
        let mut req = create_test_request("llama2");
        req.max_tokens = Some(256);

        let ollama_req = to_ollama_request(&req, false);

        assert!(ollama_req.options.is_some());
        let opts = ollama_req.options.unwrap();
        assert_eq!(opts.num_predict, Some(256));
    }

    #[test]
    fn test_options_mapping_all_params() {
        let mut req = create_test_request("llama2");
        req.temperature = Some(0.7);
        req.top_p = Some(0.9);
        req.max_tokens = Some(512);
        req.stop = Some(vec!["END".to_string(), "DONE".to_string()]);

        let ollama_req = to_ollama_request(&req, false);

        assert!(ollama_req.options.is_some());
        let opts = ollama_req.options.unwrap();
        assert_eq!(opts.temperature, Some(0.7));
        assert_eq!(opts.top_p, Some(0.9));
        assert_eq!(opts.num_predict, Some(512));
        assert_eq!(opts.stop, Some(vec!["END".to_string(), "DONE".to_string()]));
    }

    #[test]
    fn test_options_none_when_no_params() {
        let req = create_test_request("llama2");

        let ollama_req = to_ollama_request(&req, false);

        assert!(ollama_req.options.is_none());
    }

    #[test]
    fn test_options_present_only_with_temperature() {
        let mut req = create_test_request("llama2");
        req.temperature = Some(0.5);
        req.max_tokens = None;
        req.top_p = None;
        req.stop = None;

        let ollama_req = to_ollama_request(&req, false);

        assert!(ollama_req.options.is_some());
    }

    #[test]
    fn test_options_present_only_with_stop() {
        let mut req = create_test_request("llama2");
        req.temperature = None;
        req.max_tokens = None;
        req.top_p = None;
        req.stop = Some(vec!["###".to_string()]);

        let ollama_req = to_ollama_request(&req, false);

        assert!(ollama_req.options.is_some());
        let opts = ollama_req.options.unwrap();
        assert_eq!(opts.stop, Some(vec!["###".to_string()]));
        assert_eq!(opts.temperature, None);
    }

    #[test]
    fn test_response_format_json_object() {
        let mut req = create_test_request("llama2");
        req.response_format = Some(ResponseFormat {
            format_type: ResponseFormatType::JsonObject,
            json_schema: None,
        });

        let ollama_req = to_ollama_request(&req, false);

        assert!(ollama_req.format.is_some());
        let format_val = ollama_req.format.unwrap();
        assert_eq!(format_val, serde_json::Value::String("json".to_string()));
    }

    #[test]
    fn test_response_format_json_schema_with_schema() {
        let mut req = create_test_request("llama2");
        let schema_obj = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer" }
            }
        });
        req.response_format = Some(ResponseFormat {
            format_type: ResponseFormatType::JsonSchema,
            json_schema: Some(JsonSchemaFormat {
                name: "Person".to_string(),
                description: Some("A person's info".to_string()),
                schema: Some(schema_obj.clone()),
                strict: Some(true),
            }),
        });

        let ollama_req = to_ollama_request(&req, false);

        assert!(ollama_req.format.is_some());
        let format_val = ollama_req.format.unwrap();
        assert_eq!(format_val, schema_obj);
    }

    #[test]
    fn test_response_format_json_schema_without_schema_defaults_to_json() {
        let mut req = create_test_request("llama2");
        req.response_format = Some(ResponseFormat {
            format_type: ResponseFormatType::JsonSchema,
            json_schema: Some(JsonSchemaFormat {
                name: "Empty".to_string(),
                description: None,
                schema: None,
                strict: None,
            }),
        });

        let ollama_req = to_ollama_request(&req, false);

        assert!(ollama_req.format.is_some());
        let format_val = ollama_req.format.unwrap();
        assert_eq!(format_val, serde_json::Value::String("json".to_string()));
    }

    #[test]
    fn test_response_format_json_schema_no_schema_object() {
        let mut req = create_test_request("llama2");
        req.response_format = Some(ResponseFormat {
            format_type: ResponseFormatType::JsonSchema,
            json_schema: None,
        });

        let ollama_req = to_ollama_request(&req, false);

        assert!(ollama_req.format.is_none());
    }

    #[test]
    fn test_response_format_text() {
        let mut req = create_test_request("llama2");
        req.response_format = Some(ResponseFormat {
            format_type: ResponseFormatType::Text,
            json_schema: None,
        });

        let ollama_req = to_ollama_request(&req, false);

        assert!(ollama_req.format.is_none());
    }

    #[test]
    fn test_stream_flag_false() {
        let req = create_test_request("llama2");

        let ollama_req = to_ollama_request(&req, false);

        assert_eq!(ollama_req.stream, false);
    }

    #[test]
    fn test_stream_flag_true() {
        let req = create_test_request("llama2");

        let ollama_req = to_ollama_request(&req, true);

        assert_eq!(ollama_req.stream, true);
    }

    #[test]
    fn test_model_passthrough() {
        let req = create_test_request("custom-model:7b");

        let ollama_req = to_ollama_request(&req, false);

        assert_eq!(ollama_req.model, "custom-model:7b");
    }

    #[test]
    fn test_combined_options_and_format() {
        let mut req = create_test_request("llama2");
        req.temperature = Some(0.8);
        req.max_tokens = Some(1024);
        req.response_format = Some(ResponseFormat {
            format_type: ResponseFormatType::JsonObject,
            json_schema: None,
        });

        let ollama_req = to_ollama_request(&req, true);

        assert_eq!(ollama_req.stream, true);
        assert!(ollama_req.options.is_some());
        let opts = ollama_req.options.unwrap();
        assert_eq!(opts.temperature, Some(0.8));
        assert_eq!(opts.num_predict, Some(1024));
        assert_eq!(
            ollama_req.format,
            Some(serde_json::Value::String("json".to_string()))
        );
    }
}
