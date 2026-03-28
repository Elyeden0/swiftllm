use async_trait::async_trait;
use futures::stream::BoxStream;
use hmac::{Hmac, Mac};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::{debug, error};

use super::types::{ChatRequest, ChatResponse, FunctionCall, StreamChunk, ToolCall, Usage};
use super::{Provider, ProviderError};

type HmacSha256 = Hmac<Sha256>;

/// AWS Bedrock provider — uses the Converse API with SigV4 signing.
pub struct BedrockProvider {
    client: Client,
    region: String,
    access_key_id: String,
    secret_access_key: String,
    session_token: Option<String>,
}

impl BedrockProvider {
    pub fn new(
        region: String,
        access_key_id: String,
        secret_access_key: String,
        session_token: Option<String>,
    ) -> Self {
        Self {
            client: Client::new(),
            region,
            access_key_id,
            secret_access_key,
            session_token,
        }
    }
}

// ── Bedrock Converse API types ─────────────────────────────────────────────

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ConverseRequest {
    messages: Vec<ConverseMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<Vec<SystemBlock>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inference_config: Option<InferenceConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_config: Option<ToolConfig>,
}

#[derive(Debug, Serialize)]
struct SystemBlock {
    text: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ConverseMessage {
    role: String,
    content: Vec<ContentBlock>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
enum ContentBlock {
    Text(String),
    #[serde(rename_all = "camelCase")]
    ToolUse {
        tool_use_id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename_all = "camelCase")]
    ToolResult {
        tool_use_id: String,
        content: Vec<ToolResultContent>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
enum ToolResultContent {
    #[serde(rename = "text")]
    Text(String),
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct InferenceConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct ToolConfig {
    tools: Vec<BedrockTool>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BedrockTool {
    tool_spec: BedrockToolSpec,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BedrockToolSpec {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    input_schema: BedrockInputSchema,
}

#[derive(Debug, Serialize)]
struct BedrockInputSchema {
    json: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct ConverseResponse {
    output: ConverseOutput,
    #[serde(default)]
    usage: Option<ConverseUsage>,
    #[serde(default, rename = "stopReason")]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ConverseOutput {
    message: Option<ConverseMessage>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ConverseUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    total_tokens: u64,
}

// ── Format translation ─────────────────────────────────────────────────────

fn to_converse_request(req: &ChatRequest) -> ConverseRequest {
    let mut system = Vec::new();
    let mut messages = Vec::new();

    for msg in &req.messages {
        match msg.role.as_str() {
            "system" => {
                if let Some(text) = &msg.content {
                    system.push(SystemBlock { text: text.clone() });
                }
            }
            "tool" => {
                // Tool results need to be appended to the last user message
                // or create a new user message with tool_result content
                let tool_use_id = msg.tool_call_id.clone().unwrap_or_default();
                let content = msg.content.clone().unwrap_or_default();
                messages.push(ConverseMessage {
                    role: "user".to_string(),
                    content: vec![ContentBlock::ToolResult {
                        tool_use_id,
                        content: vec![ToolResultContent::Text(content)],
                    }],
                });
            }
            "assistant" => {
                let mut content = Vec::new();
                if let Some(text) = &msg.content {
                    if !text.is_empty() {
                        content.push(ContentBlock::Text(text.clone()));
                    }
                }
                if let Some(tool_calls) = &msg.tool_calls {
                    for tc in tool_calls {
                        let input: serde_json::Value =
                            serde_json::from_str(&tc.function.arguments).unwrap_or_default();
                        content.push(ContentBlock::ToolUse {
                            tool_use_id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            input,
                        });
                    }
                }
                if !content.is_empty() {
                    messages.push(ConverseMessage {
                        role: "assistant".to_string(),
                        content,
                    });
                }
            }
            _ => {
                // user messages
                let text = msg.content.clone().unwrap_or_default();
                messages.push(ConverseMessage {
                    role: "user".to_string(),
                    content: vec![ContentBlock::Text(text)],
                });
            }
        }
    }

    let inference_config =
        if req.temperature.is_some() || req.max_tokens.is_some() || req.top_p.is_some() {
            Some(InferenceConfig {
                max_tokens: req.max_tokens,
                temperature: req.temperature,
                top_p: req.top_p,
                stop_sequences: req.stop.clone(),
            })
        } else {
            None
        };

    let tool_config = req.tools.as_ref().map(|tools| ToolConfig {
        tools: tools
            .iter()
            .map(|t| BedrockTool {
                tool_spec: BedrockToolSpec {
                    name: t.function.name.clone(),
                    description: t.function.description.clone(),
                    input_schema: BedrockInputSchema {
                        json: t
                            .function
                            .parameters
                            .clone()
                            .unwrap_or(serde_json::json!({"type": "object", "properties": {}})),
                    },
                },
            })
            .collect(),
    });

    ConverseRequest {
        messages,
        system: if system.is_empty() {
            None
        } else {
            Some(system)
        },
        inference_config,
        tool_config,
    }
}

fn converse_to_chat_response(resp: ConverseResponse, model: &str) -> ChatResponse {
    let mut content_text = String::new();
    let mut tool_calls = Vec::new();

    if let Some(msg) = &resp.output.message {
        for block in &msg.content {
            match block {
                ContentBlock::Text(t) => content_text.push_str(t),
                ContentBlock::ToolUse {
                    tool_use_id,
                    name,
                    input,
                } => {
                    tool_calls.push(ToolCall {
                        id: tool_use_id.clone(),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: name.clone(),
                            arguments: serde_json::to_string(input).unwrap_or_default(),
                        },
                    });
                }
                _ => {}
            }
        }
    }

    let usage = resp.usage.map(|u| Usage {
        prompt_tokens: u.input_tokens,
        completion_tokens: u.output_tokens,
        total_tokens: u.total_tokens,
    });

    let finish_reason = resp.stop_reason.as_deref().map(|r| match r {
        "end_turn" => "stop",
        "max_tokens" => "length",
        "tool_use" => "tool_calls",
        "stop_sequence" => "stop",
        other => other,
    });

    if !tool_calls.is_empty() {
        let mut resp = ChatResponse::new_tool_call(model.to_string(), tool_calls, usage);
        if let Some(reason) = finish_reason {
            if let Some(choice) = resp.choices.first_mut() {
                choice.finish_reason = Some(reason.to_string());
            }
        }
        resp
    } else {
        let mut resp = ChatResponse::new(model.to_string(), content_text, usage);
        if let Some(reason) = finish_reason {
            if let Some(choice) = resp.choices.first_mut() {
                choice.finish_reason = Some(reason.to_string());
            }
        }
        resp
    }
}

// ── AWS SigV4 signing ──────────────────────────────────────────────────────

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex_encode(&hasher.finalize())
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC key size");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

fn sign_request(
    method: &str,
    uri: &str,
    body: &[u8],
    region: &str,
    access_key: &str,
    secret_key: &str,
    session_token: Option<&str>,
) -> Vec<(String, String)> {
    let service = "bedrock";
    let now = chrono_now();
    let date = &now[..8];
    let payload_hash = sha256_hex(body);

    let host = format!("bedrock-runtime.{}.amazonaws.com", region);

    let mut headers_to_sign = vec![
        ("content-type".to_string(), "application/json".to_string()),
        ("host".to_string(), host.clone()),
        ("x-amz-date".to_string(), now.clone()),
    ];
    if let Some(token) = session_token {
        headers_to_sign.push(("x-amz-security-token".to_string(), token.to_string()));
    }
    headers_to_sign.sort_by(|a, b| a.0.cmp(&b.0));

    let signed_headers: String = headers_to_sign
        .iter()
        .map(|(k, _)| k.as_str())
        .collect::<Vec<_>>()
        .join(";");

    let canonical_headers: String = headers_to_sign
        .iter()
        .map(|(k, v)| format!("{}:{}\n", k, v))
        .collect();

    let canonical_request = format!(
        "{}\n{}\n\n{}\n{}\n{}",
        method, uri, canonical_headers, signed_headers, payload_hash
    );

    let credential_scope = format!("{}/{}/{}/aws4_request", date, region, service);
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{}\n{}\n{}",
        now,
        credential_scope,
        sha256_hex(canonical_request.as_bytes())
    );

    let k_date = hmac_sha256(format!("AWS4{}", secret_key).as_bytes(), date.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, service.as_bytes());
    let k_signing = hmac_sha256(&k_service, b"aws4_request");
    let signature = hex_encode(&hmac_sha256(&k_signing, string_to_sign.as_bytes()));

    let auth_header = format!(
        "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
        access_key, credential_scope, signed_headers, signature
    );

    let mut result = vec![
        ("Authorization".to_string(), auth_header),
        ("x-amz-date".to_string(), now),
        ("Content-Type".to_string(), "application/json".to_string()),
    ];
    if let Some(token) = session_token {
        result.push(("x-amz-security-token".to_string(), token.to_string()));
    }
    result
}

/// ISO 8601 basic format: 20060102T150405Z
fn chrono_now() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    // Convert unix timestamp to YYYYMMDDTHHmmssZ
    let secs_per_day = 86400u64;
    let days = now / secs_per_day;
    let secs_today = now % secs_per_day;
    let hours = secs_today / 3600;
    let minutes = (secs_today % 3600) / 60;
    let seconds = secs_today % 60;

    // Simple days-to-date calculation
    let (year, month, day) = days_to_ymd(days);
    format!(
        "{:04}{:02}{:02}T{:02}{:02}{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    // Days since 1970-01-01
    let mut year = 1970u64;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    let months_days: [u64; 12] = if is_leap(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 1u64;
    for &md in &months_days {
        if days < md {
            break;
        }
        days -= md;
        month += 1;
    }
    (year, month, days + 1)
}

fn is_leap(y: u64) -> bool {
    (y.is_multiple_of(4) && !y.is_multiple_of(100)) || y.is_multiple_of(400)
}

// ── Provider implementation ────────────────────────────────────────────────

#[async_trait]
impl Provider for BedrockProvider {
    fn name(&self) -> &str {
        "bedrock"
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        debug!("Bedrock chat request for model: {}", request.model);

        let uri = format!("/model/{}/converse", request.model);
        let url = format!(
            "https://bedrock-runtime.{}.amazonaws.com{}",
            self.region, uri
        );

        let converse_req = to_converse_request(request);
        let body =
            serde_json::to_vec(&converse_req).map_err(|e| ProviderError::Parse(e.to_string()))?;

        let headers = sign_request(
            "POST",
            &uri,
            &body,
            &self.region,
            &self.access_key_id,
            &self.secret_access_key,
            self.session_token.as_deref(),
        );

        let mut builder = self.client.post(&url);
        for (k, v) in &headers {
            builder = builder.header(k.as_str(), v.as_str());
        }
        builder = builder.body(body);

        let response = builder
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        let status = response.status().as_u16();
        if status >= 400 {
            let body = response.text().await.unwrap_or_default();
            error!("Bedrock API error {}: {}", status, body);
            return Err(ProviderError::Api {
                status,
                message: body,
            });
        }

        let converse_resp: ConverseResponse = response
            .json()
            .await
            .map_err(|e| ProviderError::Parse(e.to_string()))?;

        Ok(converse_to_chat_response(converse_resp, &request.model))
    }

    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, ProviderError>>, ProviderError> {
        debug!(
            "Bedrock streaming request for model: {} (falling back to non-streaming)",
            request.model
        );

        // Bedrock ConverseStream has a complex event-stream format.
        // For simplicity, we fall back to a single-chunk stream from the non-streaming API.
        let response = self.chat(request).await?;

        let model = request.model.clone();
        let content = response
            .choices
            .first()
            .and_then(|c| c.message.content.clone());
        let finish_reason = response
            .choices
            .first()
            .and_then(|c| c.finish_reason.clone());

        let chunk = StreamChunk::new(&model, content, finish_reason);
        let stream = futures::stream::once(async move { Ok(chunk) });

        Ok(Box::pin(stream))
    }
}
