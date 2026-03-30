use serde::{Deserialize, Serialize};

// ── Tool / Function-calling types ──────────────────────────────────────────

/// A tool the model may call (OpenAI-compatible format).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String, // always "function"
    pub function: FunctionDefinition,
}

/// Description of a callable function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// A tool call emitted by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String, // "function"
    pub function: FunctionCall,
}

/// The function the model wants to invoke.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String, // JSON string
}

/// A tool call delta for streaming responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

// ── Structured output / response format ────────────────────────────────────

/// The type of response format requested.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormatType {
    Text,
    JsonObject,
    JsonSchema,
}

/// A JSON Schema definition used with `json_schema` response format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchemaFormat {
    /// A name for the schema (required by OpenAI).
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The JSON Schema object itself.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<serde_json::Value>,
    /// Whether to enable strict schema adherence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// OpenAI-compatible `response_format` parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: ResponseFormatType,
    /// Present only when `format_type` is `json_schema`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<JsonSchemaFormat>,
}

// ── Chat completion request ────────────────────────────────────────────────

/// OpenAI-compatible chat completion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    /// Tool definitions the model may call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    /// Controls tool usage: "auto", "none", "required", or a specific function.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    /// Structured output / JSON mode configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Tool calls requested by the assistant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// ID of the tool call this message is responding to (role = "tool").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Function name (used with role = "tool").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

// ── Chat completion response ───────────────────────────────────────────────

/// OpenAI-compatible chat completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

// ── Streaming types ────────────────────────────────────────────────────────

/// OpenAI-compatible streaming chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

// ── Embedding types ────────────────────────────────────────────────────────

/// Input for an embedding request — either a single string or an array of strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

impl EmbeddingInput {
    /// Return the inputs as a slice-friendly list.
    pub fn to_vec(&self) -> Vec<String> {
        match self {
            EmbeddingInput::Single(s) => vec![s.clone()],
            EmbeddingInput::Multiple(v) => v.clone(),
        }
    }
}

/// OpenAI-compatible embedding request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u64>,
}

/// OpenAI-compatible embedding response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

/// A single embedding result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f64>,
    pub index: usize,
}

/// Token usage for embedding requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u64,
    pub total_tokens: u64,
}

impl EmbeddingResponse {
    pub fn new(model: String, data: Vec<EmbeddingData>, usage: EmbeddingUsage) -> Self {
        Self {
            object: "list".to_string(),
            data,
            model,
            usage,
        }
    }
}

// ── Constructors ───────────────────────────────────────────────────────────

impl ChatRequest {
    pub fn is_streaming(&self) -> bool {
        self.stream.unwrap_or(false)
    }
}

impl ChatResponse {
    pub fn new(model: String, content: String, usage: Option<Usage>) -> Self {
        Self {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            model,
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: Some(content),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage,
        }
    }

    /// Create a response that contains tool calls instead of text content.
    pub fn new_tool_call(model: String, tool_calls: Vec<ToolCall>, usage: Option<Usage>) -> Self {
        Self {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            model,
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(tool_calls),
                    tool_call_id: None,
                    name: None,
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage,
        }
    }
}

impl StreamChunk {
    pub fn new(model: &str, content: Option<String>, finish_reason: Option<String>) -> Self {
        Self {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion.chunk".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            model: model.to_string(),
            choices: vec![StreamChoice {
                index: 0,
                delta: Delta {
                    role: if content.is_none() && finish_reason.is_none() {
                        Some("assistant".to_string())
                    } else {
                        None
                    },
                    content,
                    tool_calls: None,
                },
                finish_reason,
            }],
        }
    }

    /// Create a stream chunk that carries a tool-call delta.
    pub fn new_tool_call_delta(
        model: &str,
        tool_calls: Vec<ToolCallDelta>,
        finish_reason: Option<String>,
    ) -> Self {
        Self {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion.chunk".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            model: model.to_string(),
            choices: vec![StreamChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: None,
                    tool_calls: Some(tool_calls),
                },
                finish_reason,
            }],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_input_single_to_vec() {
        let input = EmbeddingInput::Single("hello".to_string());
        assert_eq!(input.to_vec(), vec!["hello".to_string()]);
    }

    #[test]
    fn test_embedding_input_multiple_to_vec() {
        let input = EmbeddingInput::Multiple(vec!["hello".to_string(), "world".to_string()]);
        assert_eq!(
            input.to_vec(),
            vec!["hello".to_string(), "world".to_string()]
        );
    }

    #[test]
    fn test_embedding_input_deserialize_single() {
        let json = r#""hello world""#;
        let input: EmbeddingInput = serde_json::from_str(json).unwrap();
        match input {
            EmbeddingInput::Single(s) => assert_eq!(s, "hello world"),
            _ => panic!("Expected Single variant"),
        }
    }

    #[test]
    fn test_embedding_input_deserialize_multiple() {
        let json = r#"["hello", "world"]"#;
        let input: EmbeddingInput = serde_json::from_str(json).unwrap();
        match input {
            EmbeddingInput::Multiple(v) => {
                assert_eq!(v, vec!["hello".to_string(), "world".to_string()])
            }
            _ => panic!("Expected Multiple variant"),
        }
    }

    #[test]
    fn test_embedding_request_deserialize() {
        let json = r#"{
            "model": "text-embedding-3-small",
            "input": "The quick brown fox"
        }"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "text-embedding-3-small");
        assert!(req.encoding_format.is_none());
        assert!(req.dimensions.is_none());
    }

    #[test]
    fn test_embedding_request_with_options() {
        let json = r#"{
            "model": "text-embedding-3-large",
            "input": ["hello", "world"],
            "encoding_format": "float",
            "dimensions": 256
        }"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "text-embedding-3-large");
        assert_eq!(req.encoding_format, Some("float".to_string()));
        assert_eq!(req.dimensions, Some(256));
    }

    #[test]
    fn test_embedding_response_new() {
        let data = vec![EmbeddingData {
            object: "embedding".to_string(),
            embedding: vec![0.1, 0.2, 0.3],
            index: 0,
        }];
        let usage = EmbeddingUsage {
            prompt_tokens: 5,
            total_tokens: 5,
        };
        let resp = EmbeddingResponse::new("test-model".to_string(), data, usage);
        assert_eq!(resp.object, "list");
        assert_eq!(resp.model, "test-model");
        assert_eq!(resp.data.len(), 1);
        assert_eq!(resp.data[0].embedding, vec![0.1, 0.2, 0.3]);
        assert_eq!(resp.usage.prompt_tokens, 5);
    }

    #[test]
    fn test_embedding_response_serialize() {
        let resp = EmbeddingResponse::new(
            "text-embedding-3-small".to_string(),
            vec![EmbeddingData {
                object: "embedding".to_string(),
                embedding: vec![0.5, -0.5],
                index: 0,
            }],
            EmbeddingUsage {
                prompt_tokens: 3,
                total_tokens: 3,
            },
        );
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["model"], "text-embedding-3-small");
        assert_eq!(json["data"][0]["object"], "embedding");
        assert_eq!(json["data"][0]["index"], 0);
    }
}
