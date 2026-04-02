//! Node.js/TypeScript bindings for SwiftLLM via NAPI-RS.
//!
//! Provides a native Node.js addon that calls LLM provider APIs directly.
//! Supports OpenAI, Anthropic, Gemini, and Mistral through a unified API.

use std::collections::HashMap;
use std::sync::Mutex;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A chat message.
#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Token usage statistics.
#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
}

/// A tool call returned by the model.
#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub function_name: String,
    pub function_arguments: String,
}

/// Response from a chat completion call.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub id: String,
    pub model: String,
    pub content: Option<String>,
    pub finish_reason: Option<String>,
    pub usage: Option<Usage>,
    pub tool_calls: Option<Vec<ToolCall>>,
    /// The full raw JSON response as a string.
    pub raw: String,
}

/// Options for a completion request.
#[napi(object)]
#[derive(Debug, Clone, Default)]
pub struct CompletionOptions {
    pub temperature: Option<f64>,
    pub max_tokens: Option<i64>,
    pub top_p: Option<f64>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
}

/// Options when adding a provider.
#[napi(object)]
#[derive(Debug, Clone, Default)]
pub struct ProviderOptions {
    pub base_url: Option<String>,
    pub models: Option<Vec<String>>,
}

// ---------------------------------------------------------------------------
// Internal provider types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ProviderInfo {
    kind: ProviderKindInternal,
    api_key: String,
    base_url: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
enum ProviderKindInternal {
    Openai,
    Anthropic,
    Gemini,
    Mistral,
}

// ---------------------------------------------------------------------------
// OpenAI-compatible request/response shapes (used for direct HTTP calls)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAiMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    id: String,
    model: String,
    choices: Vec<OpenAiChoice>,
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiChoiceMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoiceMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAiToolCall {
    id: String,
    function: OpenAiFunction,
}

#[derive(Debug, Deserialize)]
struct OpenAiFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    prompt_tokens: i64,
    completion_tokens: i64,
    total_tokens: i64,
}

// ---------------------------------------------------------------------------
// Anthropic request/response shapes
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    id: String,
    model: String,
    content: Vec<AnthropicContent>,
    usage: AnthropicUsage,
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    _content_type: String,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: i64,
    output_tokens: i64,
}

// ---------------------------------------------------------------------------
// Gemini request/response shapes
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "generationConfig")]
    generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Debug, Serialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize)]
struct GeminiPart {
    text: String,
}

#[derive(Debug, Serialize)]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "maxOutputTokens")]
    max_output_tokens: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "topP")]
    top_p: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: GeminiCandidateContent,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidateContent {
    parts: Option<Vec<GeminiResponsePart>>,
}

#[derive(Debug, Deserialize)]
struct GeminiResponsePart {
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiUsageMetadata {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: Option<i64>,
    #[serde(rename = "candidatesTokenCount")]
    candidates_token_count: Option<i64>,
    #[serde(rename = "totalTokenCount")]
    total_token_count: Option<i64>,
}

// ---------------------------------------------------------------------------
// API error response
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    error: Option<ApiErrorDetail>,
}

#[derive(Debug, Deserialize)]
struct ApiErrorDetail {
    message: Option<String>,
}

// ---------------------------------------------------------------------------
// Provider dispatch
// ---------------------------------------------------------------------------

fn infer_provider(model: &str) -> Result<ProviderKindInternal> {
    let m = model.to_lowercase();
    if m.starts_with("gpt-")
        || m.starts_with("o1")
        || m.starts_with("o3")
        || m.starts_with("o4")
        || m.starts_with("chatgpt-")
    {
        Ok(ProviderKindInternal::Openai)
    } else if m.starts_with("claude-") {
        Ok(ProviderKindInternal::Anthropic)
    } else if m.starts_with("gemini-") {
        Ok(ProviderKindInternal::Gemini)
    } else if m.starts_with("mistral-")
        || m.starts_with("codestral")
        || m.starts_with("pixtral")
        || m.starts_with("ministral")
        || m.starts_with("open-mistral")
    {
        Ok(ProviderKindInternal::Mistral)
    } else {
        Err(Error::new(
            Status::InvalidArg,
            format!(
                "Cannot infer provider for model '{}'. Use SwiftLLM class to register a provider.",
                model
            ),
        ))
    }
}

fn default_base_url(kind: &ProviderKindInternal) -> &'static str {
    match kind {
        ProviderKindInternal::Openai => "https://api.openai.com",
        ProviderKindInternal::Anthropic => "https://api.anthropic.com",
        ProviderKindInternal::Gemini => "https://generativelanguage.googleapis.com",
        ProviderKindInternal::Mistral => "https://api.mistral.ai",
    }
}

fn parse_kind(name: &str) -> Result<ProviderKindInternal> {
    match name.to_lowercase().as_str() {
        "openai" => Ok(ProviderKindInternal::Openai),
        "anthropic" => Ok(ProviderKindInternal::Anthropic),
        "gemini" => Ok(ProviderKindInternal::Gemini),
        "mistral" => Ok(ProviderKindInternal::Mistral),
        other => Err(Error::new(
            Status::InvalidArg,
            format!(
                "Unknown provider: '{}'. Expected: openai, anthropic, gemini, mistral",
                other
            ),
        )),
    }
}

fn messages_to_openai(messages: &[Message]) -> Vec<OpenAiMessage> {
    messages
        .iter()
        .map(|m| OpenAiMessage {
            role: m.role.clone(),
            content: m.content.clone(),
            name: m.name.clone(),
            tool_call_id: m.tool_call_id.clone(),
        })
        .collect()
}

async fn call_openai_compatible(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    messages: &[Message],
    opts: &CompletionOptions,
) -> Result<CompletionResponse> {
    let url = format!("{}/v1/chat/completions", base_url);
    let body = OpenAiRequest {
        model: model.to_string(),
        messages: messages_to_openai(messages),
        temperature: opts.temperature,
        max_tokens: opts.max_tokens,
        top_p: opts.top_p,
    };

    let response = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Network error: {}", e)))?;

    let status = response.status().as_u16();
    let raw_text = response
        .text()
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Read error: {}", e)))?;

    if status >= 400 {
        let msg = serde_json::from_str::<ApiErrorResponse>(&raw_text)
            .ok()
            .and_then(|e| e.error)
            .and_then(|e| e.message)
            .unwrap_or_else(|| raw_text.clone());
        return Err(Error::new(
            Status::GenericFailure,
            format!("API error ({}): {}", status, msg),
        ));
    }

    let resp: OpenAiResponse = serde_json::from_str(&raw_text)
        .map_err(|e| Error::new(Status::GenericFailure, format!("Parse error: {}", e)))?;

    let choice = resp.choices.first();
    let content = choice.and_then(|c| c.message.content.clone());
    let finish_reason = choice.and_then(|c| c.finish_reason.clone());
    let tool_calls = choice.and_then(|c| {
        c.message.tool_calls.as_ref().map(|tcs| {
            tcs.iter()
                .map(|tc| ToolCall {
                    id: tc.id.clone(),
                    function_name: tc.function.name.clone(),
                    function_arguments: tc.function.arguments.clone(),
                })
                .collect()
        })
    });
    let usage = resp.usage.map(|u| Usage {
        prompt_tokens: u.prompt_tokens,
        completion_tokens: u.completion_tokens,
        total_tokens: u.total_tokens,
    });

    Ok(CompletionResponse {
        id: resp.id,
        model: resp.model,
        content,
        finish_reason,
        usage,
        tool_calls,
        raw: raw_text,
    })
}

async fn call_anthropic(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    messages: &[Message],
    opts: &CompletionOptions,
) -> Result<CompletionResponse> {
    let url = format!("{}/v1/messages", base_url);

    // Extract system message
    let system = messages
        .iter()
        .find(|m| m.role == "system")
        .and_then(|m| m.content.clone());

    let anthropic_messages: Vec<AnthropicMessage> = messages
        .iter()
        .filter(|m| m.role != "system")
        .map(|m| AnthropicMessage {
            role: m.role.clone(),
            content: m.content.clone().unwrap_or_default(),
        })
        .collect();

    let body = AnthropicRequest {
        model: model.to_string(),
        max_tokens: opts.max_tokens.unwrap_or(4096),
        system,
        messages: anthropic_messages,
        temperature: opts.temperature,
        top_p: opts.top_p,
    };

    let response = client
        .post(&url)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Network error: {}", e)))?;

    let status = response.status().as_u16();
    let raw_text = response
        .text()
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Read error: {}", e)))?;

    if status >= 400 {
        let msg = serde_json::from_str::<ApiErrorResponse>(&raw_text)
            .ok()
            .and_then(|e| e.error)
            .and_then(|e| e.message)
            .unwrap_or_else(|| raw_text.clone());
        return Err(Error::new(
            Status::GenericFailure,
            format!("API error ({}): {}", status, msg),
        ));
    }

    let resp: AnthropicResponse = serde_json::from_str(&raw_text)
        .map_err(|e| Error::new(Status::GenericFailure, format!("Parse error: {}", e)))?;

    let content = resp
        .content
        .iter()
        .filter_map(|c| c.text.clone())
        .collect::<Vec<_>>()
        .join("");
    let content = if content.is_empty() {
        None
    } else {
        Some(content)
    };

    let finish_reason = resp.stop_reason.map(|r| match r.as_str() {
        "end_turn" => "stop".to_string(),
        "max_tokens" => "length".to_string(),
        other => other.to_string(),
    });

    let total = resp.usage.input_tokens + resp.usage.output_tokens;

    Ok(CompletionResponse {
        id: resp.id,
        model: resp.model,
        content,
        finish_reason,
        usage: Some(Usage {
            prompt_tokens: resp.usage.input_tokens,
            completion_tokens: resp.usage.output_tokens,
            total_tokens: total,
        }),
        tool_calls: None,
        raw: raw_text,
    })
}

async fn call_gemini(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    messages: &[Message],
    opts: &CompletionOptions,
) -> Result<CompletionResponse> {
    let url = format!(
        "{}/v1beta/models/{}:generateContent?key={}",
        base_url, model, api_key
    );

    let contents: Vec<GeminiContent> = messages
        .iter()
        .filter(|m| m.role != "system")
        .map(|m| GeminiContent {
            role: if m.role == "assistant" {
                "model".to_string()
            } else {
                "user".to_string()
            },
            parts: vec![GeminiPart {
                text: m.content.clone().unwrap_or_default(),
            }],
        })
        .collect();

    let generation_config =
        if opts.temperature.is_some() || opts.max_tokens.is_some() || opts.top_p.is_some() {
            Some(GeminiGenerationConfig {
                temperature: opts.temperature,
                max_output_tokens: opts.max_tokens,
                top_p: opts.top_p,
            })
        } else {
            None
        };

    let body = GeminiRequest {
        contents,
        generation_config,
    };

    let response = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Network error: {}", e)))?;

    let status = response.status().as_u16();
    let raw_text = response
        .text()
        .await
        .map_err(|e| Error::new(Status::GenericFailure, format!("Read error: {}", e)))?;

    if status >= 400 {
        return Err(Error::new(
            Status::GenericFailure,
            format!("API error ({}): {}", status, raw_text),
        ));
    }

    let resp: GeminiResponse = serde_json::from_str(&raw_text)
        .map_err(|e| Error::new(Status::GenericFailure, format!("Parse error: {}", e)))?;

    let candidate = resp.candidates.as_ref().and_then(|c| c.first());
    let content = candidate
        .and_then(|c| c.content.parts.as_ref())
        .and_then(|parts| {
            let text: String = parts.iter().filter_map(|p| p.text.clone()).collect();
            if text.is_empty() {
                None
            } else {
                Some(text)
            }
        });
    let finish_reason = candidate.and_then(|c| c.finish_reason.clone()).map(|r| {
        match r.as_str() {
            "STOP" => "stop".to_string(),
            "MAX_TOKENS" => "length".to_string(),
            other => other.to_lowercase(),
        }
    });

    let usage = resp.usage_metadata.map(|u| Usage {
        prompt_tokens: u.prompt_token_count.unwrap_or(0),
        completion_tokens: u.candidates_token_count.unwrap_or(0),
        total_tokens: u.total_token_count.unwrap_or(0),
    });

    let id = format!("chatcmpl-gemini-{}", std::process::id());

    Ok(CompletionResponse {
        id,
        model: model.to_string(),
        content,
        finish_reason,
        usage,
        tool_calls: None,
        raw: raw_text,
    })
}

async fn dispatch_completion(
    client: &reqwest::Client,
    kind: &ProviderKindInternal,
    base_url: &str,
    api_key: &str,
    model: &str,
    messages: &[Message],
    opts: &CompletionOptions,
) -> Result<CompletionResponse> {
    match kind {
        ProviderKindInternal::Openai | ProviderKindInternal::Mistral => {
            call_openai_compatible(client, base_url, api_key, model, messages, opts).await
        }
        ProviderKindInternal::Anthropic => {
            call_anthropic(client, base_url, api_key, model, messages, opts).await
        }
        ProviderKindInternal::Gemini => {
            call_gemini(client, base_url, api_key, model, messages, opts).await
        }
    }
}

fn messages_from_prompt(prompt: Either<String, Vec<Message>>) -> Vec<Message> {
    match prompt {
        Either::A(text) => vec![Message {
            role: "user".to_string(),
            content: Some(text),
            name: None,
            tool_call_id: None,
        }],
        Either::B(msgs) => msgs,
    }
}

// ---------------------------------------------------------------------------
// SwiftLLM class
// ---------------------------------------------------------------------------

/// The main SwiftLLM client for Node.js.
///
/// ```js
/// const { SwiftLLM } = require('swiftllm');
///
/// const llm = new SwiftLLM();
/// llm.addProvider('openai', 'sk-...');
/// const resp = await llm.completion('gpt-4o-mini', 'Hello!');
/// console.log(resp.content);
/// ```
#[napi]
pub struct SwiftLLM {
    providers: Mutex<HashMap<String, ProviderInfo>>,
    client: reqwest::Client,
}

impl Default for SwiftLLM {
    fn default() -> Self {
        Self::new()
    }
}

#[napi]
impl SwiftLLM {
    /// Create a new SwiftLLM instance.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            providers: Mutex::new(HashMap::new()),
            client: reqwest::Client::new(),
        }
    }

    /// Register a provider with its API key.
    ///
    /// @param name - Provider name: "openai", "anthropic", "gemini", or "mistral"
    /// @param apiKey - The provider's API key
    /// @param options - Optional configuration (base_url, models)
    #[napi]
    pub fn add_provider(
        &self,
        name: String,
        api_key: String,
        options: Option<ProviderOptions>,
    ) -> Result<()> {
        let kind = parse_kind(&name)?;
        let opts = options.unwrap_or_default();
        let mut providers = self.providers.lock().map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Lock error: {}", e),
            )
        })?;
        providers.insert(
            name.to_lowercase(),
            ProviderInfo {
                kind,
                api_key,
                base_url: opts.base_url,
            },
        );
        Ok(())
    }

    /// List registered provider names.
    #[napi]
    pub fn list_providers(&self) -> Result<Vec<String>> {
        let providers = self.providers.lock().map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Lock error: {}", e),
            )
        })?;
        Ok(providers.keys().cloned().collect())
    }

    /// Send an async chat completion request.
    ///
    /// @param model - Model identifier (e.g. "gpt-4o-mini", "claude-sonnet-4-20250514")
    /// @param prompt - A string or array of Message objects
    /// @param options - Optional parameters (temperature, max_tokens, etc.)
    #[napi]
    pub async fn completion(
        &self,
        model: String,
        prompt: Either<String, Vec<Message>>,
        options: Option<CompletionOptions>,
    ) -> Result<CompletionResponse> {
        let messages = messages_from_prompt(prompt);
        let opts = options.unwrap_or_default();

        // Find provider
        let (kind, api_key, base_url) = {
            let providers = self.providers.lock().map_err(|e| {
                Error::new(
                    Status::GenericFailure,
                    format!("Lock error: {}", e),
                )
            })?;

            // Try to find provider by model prefix
            let inferred_kind = infer_provider(&model)?;
            let provider = providers
                .values()
                .find(|p| p.kind == inferred_kind)
                .ok_or_else(|| {
                    Error::new(
                        Status::InvalidArg,
                        format!(
                            "No provider registered for model '{}'. Call addProvider() first.",
                            model
                        ),
                    )
                })?;

            let api_key = opts
                .api_key
                .clone()
                .unwrap_or_else(|| provider.api_key.clone());
            let base_url = opts
                .base_url
                .clone()
                .or_else(|| provider.base_url.clone())
                .unwrap_or_else(|| default_base_url(&provider.kind).to_string());
            (provider.kind.clone(), api_key, base_url)
        };

        dispatch_completion(&self.client, &kind, &base_url, &api_key, &model, &messages, &opts)
            .await
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Quick one-shot async completion — infers the provider from the model name.
///
/// ```js
/// const { asyncCompletion } = require('swiftllm');
///
/// const resp = await asyncCompletion('gpt-4o-mini', 'Hello!', { apiKey: 'sk-...' });
/// console.log(resp.content);
/// ```
#[napi]
pub async fn async_completion(
    model: String,
    prompt: Either<String, Vec<Message>>,
    options: Option<CompletionOptions>,
) -> Result<CompletionResponse> {
    let messages = messages_from_prompt(prompt);
    let opts = options.unwrap_or_default();
    let kind = infer_provider(&model)?;

    let api_key = opts.api_key.clone().ok_or_else(|| {
        Error::new(
            Status::InvalidArg,
            "api_key is required for standalone completion(). Pass it in options.",
        )
    })?;
    let base_url = opts
        .base_url
        .clone()
        .unwrap_or_else(|| default_base_url(&kind).to_string());

    let client = reqwest::Client::new();
    dispatch_completion(&client, &kind, &base_url, &api_key, &model, &messages, &opts).await
}

/// Synchronous one-shot completion — infers the provider from the model name.
/// Internally runs the async version on a blocking Tokio runtime.
///
/// ```js
/// const { completion } = require('swiftllm');
///
/// const resp = completion('gpt-4o-mini', 'Hello!', { apiKey: 'sk-...' });
/// console.log(resp.content);
/// ```
#[napi]
pub fn completion(
    model: String,
    prompt: Either<String, Vec<Message>>,
    options: Option<CompletionOptions>,
) -> Result<CompletionResponse> {
    let messages = messages_from_prompt(prompt);
    let opts = options.unwrap_or_default();
    let kind = infer_provider(&model)?;

    let api_key = opts.api_key.clone().ok_or_else(|| {
        Error::new(
            Status::InvalidArg,
            "api_key is required for standalone completion(). Pass it in options.",
        )
    })?;
    let base_url = opts
        .base_url
        .clone()
        .unwrap_or_else(|| default_base_url(&kind).to_string());

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| Error::new(Status::GenericFailure, format!("Runtime error: {}", e)))?;
    let client = reqwest::Client::new();

    rt.block_on(dispatch_completion(
        &client, &kind, &base_url, &api_key, &model, &messages, &opts,
    ))
}
