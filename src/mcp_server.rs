//! MCP (Model Context Protocol) server for SwiftLLM.
//!
//! Communicates via JSON-RPC over stdio (stdin/stdout) and exposes SwiftLLM's
//! capabilities as MCP tools: chat completions, model listing, smart routing,
//! consensus queries, usage statistics, and model comparison.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use swiftllm::config::Config;
use swiftllm::consensus::ConsensusEngine;
use swiftllm::providers::types::{ChatRequest, ConsensusConfig, ConsensusStrategy, Message};
use swiftllm::routing::{QualityTier, RoutingStrategy, SmartRoutingConfig};
use swiftllm::server::AppState;

// ── JSON-RPC types ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

impl JsonRpcResponse {
    fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: Some(result),
            error: None,
        }
    }

    fn error(id: Value, code: i32, message: String) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message,
                data: None,
            }),
        }
    }
}

// ── MCP protocol types ──────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct McpServerInfo {
    name: String,
    version: String,
}

#[derive(Debug, Serialize)]
struct McpCapabilities {
    tools: McpToolsCapability,
}

#[derive(Debug, Serialize)]
struct McpToolsCapability {}

#[derive(Debug, Serialize)]
struct McpInitializeResult {
    #[serde(rename = "protocolVersion")]
    protocol_version: String,
    capabilities: McpCapabilities,
    #[serde(rename = "serverInfo")]
    server_info: McpServerInfo,
}

#[derive(Debug, Serialize)]
struct McpTool {
    name: String,
    description: String,
    #[serde(rename = "inputSchema")]
    input_schema: Value,
}

#[derive(Debug, Serialize)]
struct McpToolResult {
    content: Vec<McpContent>,
    #[serde(rename = "isError", skip_serializing_if = "Option::is_none")]
    is_error: Option<bool>,
}

#[derive(Debug, Serialize)]
struct McpContent {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

impl McpToolResult {
    fn text(text: String) -> Self {
        Self {
            content: vec![McpContent {
                content_type: "text".into(),
                text,
            }],
            is_error: None,
        }
    }

    fn error(text: String) -> Self {
        Self {
            content: vec![McpContent {
                content_type: "text".into(),
                text,
            }],
            is_error: Some(true),
        }
    }
}

// ── Tool definitions ────────────────────────────────────────────────────────

fn tool_definitions() -> Vec<McpTool> {
    vec![
        McpTool {
            name: "chat_completion".into(),
            description: "Send a chat completion request to an LLM provider".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "model": { "type": "string", "description": "Model identifier (e.g. gpt-4o, claude-sonnet-4-20250514)" },
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": { "type": "string" },
                                "content": { "type": "string" }
                            },
                            "required": ["role", "content"]
                        },
                        "description": "Array of chat messages"
                    },
                    "temperature": { "type": "number", "description": "Sampling temperature (optional)" },
                    "max_tokens": { "type": "integer", "description": "Maximum tokens to generate (optional)" }
                },
                "required": ["model", "messages"]
            }),
        },
        McpTool {
            name: "list_models".into(),
            description: "List all available models across configured providers".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        McpTool {
            name: "list_providers".into(),
            description: "Show configured LLM providers and their details".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        McpTool {
            name: "smart_route".into(),
            description: "Use cost/latency routing to pick the best model and get a completion"
                .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "quality": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Desired quality tier"
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["cost_optimized", "latency_optimized", "balanced"],
                        "description": "Routing strategy"
                    },
                    "message": { "type": "string", "description": "The user message to send" }
                },
                "required": ["quality", "strategy", "message"]
            }),
        },
        McpTool {
            name: "consensus_query".into(),
            description:
                "Query multiple models and combine their responses using a consensus strategy"
                    .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "models": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Models to query in parallel"
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["best_of", "majority", "merge"],
                        "description": "Consensus strategy"
                    },
                    "message": { "type": "string", "description": "The user message to send" }
                },
                "required": ["models", "strategy", "message"]
            }),
        },
        McpTool {
            name: "get_stats".into(),
            description:
                "Get usage statistics including total requests, tokens, cost, and cache info".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        McpTool {
            name: "compare_models".into(),
            description: "Run the same prompt against multiple models and compare responses".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "models": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Models to compare"
                    },
                    "message": { "type": "string", "description": "The user message to send" }
                },
                "required": ["models", "message"]
            }),
        },
    ]
}

// ── Tool handlers ───────────────────────────────────────────────────────────

async fn handle_chat_completion(state: &AppState, params: &Value) -> McpToolResult {
    let model = match params.get("model").and_then(|v| v.as_str()) {
        Some(m) => m.to_string(),
        None => return McpToolResult::error("Missing required parameter: model".into()),
    };

    let messages = match params.get("messages").and_then(|v| v.as_array()) {
        Some(msgs) => {
            let mut parsed = Vec::new();
            for m in msgs {
                let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                let content = m.get("content").and_then(|v| v.as_str()).unwrap_or("");
                parsed.push(Message {
                    role: role.to_string(),
                    content: Some(content.to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                });
            }
            parsed
        }
        None => return McpToolResult::error("Missing required parameter: messages".into()),
    };

    let temperature = params.get("temperature").and_then(|v| v.as_f64());
    let max_tokens = params.get("max_tokens").and_then(|v| v.as_u64());

    let request = ChatRequest {
        model,
        messages,
        temperature,
        max_tokens,
        top_p: None,
        stream: Some(false),
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        tools: None,
        tool_choice: None,
        response_format: None,
        consensus: None,
        routing: None,
    };

    let (provider_name, _) = match state.config.find_provider_for_model(&request.model) {
        Some(p) => p,
        None => {
            return McpToolResult::error(format!(
                "No provider configured for model: {}",
                request.model
            ))
        }
    };

    let provider = match state.providers.get(provider_name) {
        Some(p) => p,
        None => return McpToolResult::error("Provider not initialized".into()),
    };

    match provider.chat(&request).await {
        Ok(response) => {
            if let Some(usage) = &response.usage {
                state.cost_tracker.record_request(
                    provider_name,
                    &request.model,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                );
            }
            McpToolResult::text(serde_json::to_string_pretty(&response).unwrap_or_default())
        }
        Err(e) => McpToolResult::error(format!("Chat completion failed: {e}")),
    }
}

async fn handle_list_models(state: &AppState) -> McpToolResult {
    let mut models = Vec::new();
    for (name, provider_config) in &state.config.providers {
        for model in &provider_config.models {
            models.push(serde_json::json!({
                "provider": name,
                "model": model,
            }));
        }
    }
    McpToolResult::text(serde_json::to_string_pretty(&models).unwrap_or_default())
}

async fn handle_list_providers(state: &AppState) -> McpToolResult {
    let mut providers = Vec::new();
    for (name, provider_config) in &state.config.providers {
        providers.push(serde_json::json!({
            "name": name,
            "kind": format!("{:?}", provider_config.kind),
            "base_url": provider_config.base_url,
            "model_count": provider_config.models.len(),
        }));
    }
    McpToolResult::text(serde_json::to_string_pretty(&providers).unwrap_or_default())
}

async fn handle_smart_route(state: &AppState, params: &Value) -> McpToolResult {
    let quality = match params.get("quality").and_then(|v| v.as_str()) {
        Some("low") => QualityTier::Low,
        Some("medium") => QualityTier::Medium,
        Some("high") => QualityTier::High,
        _ => return McpToolResult::error("Invalid quality: must be low, medium, or high".into()),
    };

    let strategy = match params.get("strategy").and_then(|v| v.as_str()) {
        Some("cost_optimized") => RoutingStrategy::CostOptimized,
        Some("latency_optimized") => RoutingStrategy::LatencyOptimized,
        Some("balanced") => RoutingStrategy::Balanced,
        _ => {
            return McpToolResult::error(
                "Invalid strategy: must be cost_optimized, latency_optimized, or balanced".into(),
            )
        }
    };

    let message = match params.get("message").and_then(|v| v.as_str()) {
        Some(m) => m.to_string(),
        None => return McpToolResult::error("Missing required parameter: message".into()),
    };

    let routing_config = SmartRoutingConfig {
        strategy,
        quality,
        max_cost_per_1k_tokens: None,
    };

    let (routed_provider, routed_model, metadata) =
        match state
            .smart_router
            .select_model(&routing_config, &state.config, &state.cost_tracker)
        {
            Ok(r) => r,
            Err(e) => return McpToolResult::error(format!("Routing failed: {e}")),
        };

    let request = ChatRequest {
        model: routed_model,
        messages: vec![Message {
            role: "user".into(),
            content: Some(message),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }],
        temperature: None,
        max_tokens: None,
        top_p: None,
        stream: Some(false),
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        tools: None,
        tool_choice: None,
        response_format: None,
        consensus: None,
        routing: None,
    };

    let provider = match state.providers.get(routed_provider.as_str()) {
        Some(p) => p,
        None => return McpToolResult::error("Routed provider not initialized".into()),
    };

    match provider.chat(&request).await {
        Ok(response) => {
            if let Some(usage) = &response.usage {
                state.cost_tracker.record_request(
                    &routed_provider,
                    &request.model,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                );
            }
            let result = serde_json::json!({
                "response": serde_json::to_value(&response).unwrap_or_default(),
                "routing_metadata": serde_json::to_value(&metadata).unwrap_or_default(),
            });
            McpToolResult::text(serde_json::to_string_pretty(&result).unwrap_or_default())
        }
        Err(e) => McpToolResult::error(format!("Smart route completion failed: {e}")),
    }
}

async fn handle_consensus_query(state: &AppState, params: &Value) -> McpToolResult {
    let models = match params.get("models").and_then(|v| v.as_array()) {
        Some(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect::<Vec<_>>(),
        None => return McpToolResult::error("Missing required parameter: models".into()),
    };

    if models.is_empty() {
        return McpToolResult::error("models array must not be empty".into());
    }

    let strategy = match params.get("strategy").and_then(|v| v.as_str()) {
        Some("best_of") => ConsensusStrategy::BestOf,
        Some("majority") => ConsensusStrategy::Majority,
        Some("merge") => ConsensusStrategy::Merge,
        _ => {
            return McpToolResult::error(
                "Invalid strategy: must be best_of, majority, or merge".into(),
            )
        }
    };

    let message = match params.get("message").and_then(|v| v.as_str()) {
        Some(m) => m.to_string(),
        None => return McpToolResult::error("Missing required parameter: message".into()),
    };

    let consensus_config = ConsensusConfig {
        models,
        strategy,
        judge: None,
    };

    let request = ChatRequest {
        model: consensus_config.models[0].clone(),
        messages: vec![Message {
            role: "user".into(),
            content: Some(message),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }],
        temperature: None,
        max_tokens: None,
        top_p: None,
        stream: Some(false),
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        tools: None,
        tool_choice: None,
        response_format: None,
        consensus: None,
        routing: None,
    };

    match ConsensusEngine::execute(state, &request, &consensus_config).await {
        Ok(response) => {
            McpToolResult::text(serde_json::to_string_pretty(&response).unwrap_or_default())
        }
        Err(e) => McpToolResult::error(format!("Consensus query failed: {e}")),
    }
}

async fn handle_get_stats(state: &AppState) -> McpToolResult {
    let mut stats = serde_json::to_value(state.cost_tracker.snapshot()).unwrap_or_default();

    if let Some(cache) = &state.cache {
        stats["cache"] = serde_json::to_value(cache.stats()).unwrap_or_default();
    }

    if let Some(limiter) = &state.rate_limiter {
        stats["rate_limits"] = serde_json::to_value(limiter.stats()).unwrap_or_default();
    }

    McpToolResult::text(serde_json::to_string_pretty(&stats).unwrap_or_default())
}

async fn handle_compare_models(state: &AppState, params: &Value) -> McpToolResult {
    let models = match params.get("models").and_then(|v| v.as_array()) {
        Some(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect::<Vec<_>>(),
        None => return McpToolResult::error("Missing required parameter: models".into()),
    };

    if models.is_empty() {
        return McpToolResult::error("models array must not be empty".into());
    }

    let message = match params.get("message").and_then(|v| v.as_str()) {
        Some(m) => m.to_string(),
        None => return McpToolResult::error("Missing required parameter: message".into()),
    };

    let mut results = Vec::new();

    for model_name in &models {
        let request = ChatRequest {
            model: model_name.clone(),
            messages: vec![Message {
                role: "user".into(),
                content: Some(message.clone()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            temperature: None,
            max_tokens: None,
            top_p: None,
            stream: Some(false),
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            consensus: None,
            routing: None,
        };

        let (provider_name, _) = match state.config.find_provider_for_model(model_name) {
            Some(p) => p,
            None => {
                results.push(serde_json::json!({
                    "model": model_name,
                    "error": format!("No provider for model: {model_name}"),
                }));
                continue;
            }
        };

        let provider = match state.providers.get(provider_name) {
            Some(p) => p,
            None => {
                results.push(serde_json::json!({
                    "model": model_name,
                    "error": "Provider not initialized",
                }));
                continue;
            }
        };

        let start = std::time::Instant::now();
        match provider.chat(&request).await {
            Ok(response) => {
                let latency_ms = start.elapsed().as_millis() as u64;

                let tokens_used = response
                    .usage
                    .as_ref()
                    .map(|u| u.prompt_tokens + u.completion_tokens)
                    .unwrap_or(0);

                if let Some(usage) = &response.usage {
                    state.cost_tracker.record_request(
                        provider_name,
                        model_name,
                        usage.prompt_tokens,
                        usage.completion_tokens,
                    );
                }

                let response_text = response
                    .choices
                    .first()
                    .and_then(|c| c.message.content.as_deref())
                    .unwrap_or("");

                results.push(serde_json::json!({
                    "model": model_name,
                    "response": response_text,
                    "latency_ms": latency_ms,
                    "tokens_used": tokens_used,
                }));
            }
            Err(e) => {
                results.push(serde_json::json!({
                    "model": model_name,
                    "error": format!("{e}"),
                }));
            }
        }
    }

    McpToolResult::text(serde_json::to_string_pretty(&results).unwrap_or_default())
}

// ── Main dispatch ───────────────────────────────────────────────────────────

async fn dispatch(state: &AppState, request: JsonRpcRequest) -> JsonRpcResponse {
    let id = request.id.unwrap_or(Value::Null);

    match request.method.as_str() {
        "initialize" => {
            let result = McpInitializeResult {
                protocol_version: "2024-11-05".into(),
                capabilities: McpCapabilities {
                    tools: McpToolsCapability {},
                },
                server_info: McpServerInfo {
                    name: "swiftllm-mcp".into(),
                    version: env!("CARGO_PKG_VERSION").into(),
                },
            };
            JsonRpcResponse::success(id, serde_json::to_value(result).unwrap())
        }

        "notifications/initialized" => {
            // Client acknowledgement — no response needed for notifications,
            // but since we already have an id, respond with empty result.
            JsonRpcResponse::success(id, Value::Null)
        }

        "tools/list" => {
            let tools = tool_definitions();
            JsonRpcResponse::success(id, serde_json::json!({ "tools": tools }))
        }

        "tools/call" => {
            let tool_name = request
                .params
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let arguments = request
                .params
                .get("arguments")
                .cloned()
                .unwrap_or(Value::Object(serde_json::Map::new()));

            let result = match tool_name {
                "chat_completion" => handle_chat_completion(state, &arguments).await,
                "list_models" => handle_list_models(state).await,
                "list_providers" => handle_list_providers(state).await,
                "smart_route" => handle_smart_route(state, &arguments).await,
                "consensus_query" => handle_consensus_query(state, &arguments).await,
                "get_stats" => handle_get_stats(state).await,
                "compare_models" => handle_compare_models(state, &arguments).await,
                _ => McpToolResult::error(format!("Unknown tool: {tool_name}")),
            };

            JsonRpcResponse::success(id, serde_json::to_value(result).unwrap())
        }

        _ => JsonRpcResponse::error(id, -32601, format!("Method not found: {}", request.method)),
    }
}

// ── Entry point ─────────────────────────────────────────────────────────────

/// Find the .env file, searching next to the executable first, then the current directory.
fn find_env_file() -> Option<PathBuf> {
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let env_path = exe_dir.join(".env");
            if env_path.exists() {
                return Some(env_path);
            }
        }
    }
    let cwd_env = PathBuf::from(".env");
    if cwd_env.exists() {
        return Some(cwd_env);
    }
    None
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load env
    if let Some(env_path) = find_env_file() {
        dotenvy::from_path(&env_path)?;
    }

    // Initialize tracing to stderr (stdout is reserved for JSON-RPC)
    tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "swiftllm=info".into()),
        )
        .init();

    let config = Config::load_from_env()?;
    let state = Arc::new(AppState::new(config));

    eprintln!("swiftllm-mcp v{} ready (stdio)", env!("CARGO_PKG_VERSION"));

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Error reading stdin: {e}");
                break;
            }
        };

        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let request: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let err = JsonRpcResponse::error(Value::Null, -32700, format!("Parse error: {e}"));
                let out = serde_json::to_string(&err).unwrap();
                writeln!(stdout, "{out}")?;
                stdout.flush()?;
                continue;
            }
        };

        // Handle notifications (no id) — they don't get responses
        if request.id.is_none() && request.method.starts_with("notifications/") {
            continue;
        }

        let response = dispatch(&state, request).await;
        let out = serde_json::to_string(&response)?;
        writeln!(stdout, "{out}")?;
        stdout.flush()?;
    }

    Ok(())
}
