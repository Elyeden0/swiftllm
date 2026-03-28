//! Python bindings for SwiftLLM via PyO3.
//!
//! This module exposes the core SwiftLLM gateway as a Python-native library,
//! so users can `pip install swiftllm` and call it from Python directly.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::runtime::Runtime;

use secrecy::SecretString;

use crate::config::{
    AuthConfig, CacheConfig, Config, ProviderConfig, ProviderKind, RateLimitConfig, RoutingConfig,
};
use crate::providers::types::{ChatRequest, Message, ToolDefinition};
use crate::providers::Provider;
use crate::server::AppState;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_kind(s: &str) -> PyResult<ProviderKind> {
    match s.to_lowercase().as_str() {
        "openai" => Ok(ProviderKind::Openai),
        "anthropic" => Ok(ProviderKind::Anthropic),
        "gemini" => Ok(ProviderKind::Gemini),
        "mistral" => Ok(ProviderKind::Mistral),
        "ollama" => Ok(ProviderKind::Ollama),
        "groq" => Ok(ProviderKind::Groq),
        "together" => Ok(ProviderKind::Together),
        "bedrock" => Ok(ProviderKind::Bedrock),
        other => Err(PyValueError::new_err(format!(
            "Unknown provider kind: '{}'. Expected one of: openai, anthropic, gemini, mistral, ollama, groq, together, bedrock",
            other
        ))),
    }
}

fn provider_err_to_py(e: crate::providers::ProviderError) -> PyErr {
    PyRuntimeError::new_err(format!("{}", e))
}

// ---------------------------------------------------------------------------
// Python-visible classes
// ---------------------------------------------------------------------------

/// A single message in a chat conversation.
#[pyclass(name = "Message")]
#[derive(Clone)]
pub struct PyMessage {
    #[pyo3(get, set)]
    pub role: String,
    #[pyo3(get, set)]
    pub content: Option<String>,
}

#[pymethods]
impl PyMessage {
    #[new]
    #[pyo3(signature = (role, content=None))]
    fn new(role: String, content: Option<String>) -> Self {
        Self { role, content }
    }

    fn __repr__(&self) -> String {
        format!(
            "Message(role='{}', content={})",
            self.role,
            match &self.content {
                Some(c) => format!("'{}'", if c.len() > 60 { &c[..60] } else { c }),
                None => "None".to_string(),
            }
        )
    }
}

/// Response from a chat completion call.
#[pyclass(name = "ChatCompletionResponse")]
#[derive(Clone)]
pub struct PyChatResponse {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub model: String,
    #[pyo3(get)]
    pub content: Option<String>,
    #[pyo3(get)]
    pub finish_reason: Option<String>,
    #[pyo3(get)]
    pub prompt_tokens: Option<u64>,
    #[pyo3(get)]
    pub completion_tokens: Option<u64>,
    #[pyo3(get)]
    pub total_tokens: Option<u64>,
    #[pyo3(get)]
    pub tool_calls: Option<Vec<PyToolCall>>,
    /// The full raw JSON as a Python dict (for advanced users).
    raw_json: String,
}

/// A tool call returned by the model.
#[pyclass(name = "ToolCall")]
#[derive(Clone)]
pub struct PyToolCall {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub function_name: String,
    #[pyo3(get)]
    pub function_arguments: String,
}

#[pymethods]
impl PyToolCall {
    fn __repr__(&self) -> String {
        format!(
            "ToolCall(id='{}', function='{}', args='{}')",
            self.id,
            self.function_name,
            if self.function_arguments.len() > 60 {
                &self.function_arguments[..60]
            } else {
                &self.function_arguments
            }
        )
    }
}

#[pymethods]
impl PyChatResponse {
    #[getter]
    fn raw(&self, py: Python<'_>) -> PyResult<PyObject> {
        let module = py.import_bound("json")?;
        let result = module.call_method1("loads", (&self.raw_json,))?;
        Ok(result.into())
    }

    fn __repr__(&self) -> String {
        let preview = self
            .content
            .as_deref()
            .map(|c| if c.len() > 80 { &c[..80] } else { c })
            .unwrap_or("None");
        format!(
            "ChatCompletionResponse(model='{}', content='{}')",
            self.model, preview
        )
    }
}

impl From<crate::providers::types::ChatResponse> for PyChatResponse {
    fn from(r: crate::providers::types::ChatResponse) -> Self {
        let content = r.choices.first().and_then(|c| c.message.content.clone());
        let finish_reason = r.choices.first().and_then(|c| c.finish_reason.clone());
        let (pt, ct, tt) = r
            .usage
            .as_ref()
            .map(|u| {
                (
                    Some(u.prompt_tokens),
                    Some(u.completion_tokens),
                    Some(u.total_tokens),
                )
            })
            .unwrap_or((None, None, None));

        let tool_calls = r
            .choices
            .first()
            .and_then(|c| c.message.tool_calls.as_ref())
            .map(|tcs| {
                tcs.iter()
                    .map(|tc| PyToolCall {
                        id: tc.id.clone(),
                        function_name: tc.function.name.clone(),
                        function_arguments: tc.function.arguments.clone(),
                    })
                    .collect()
            });

        Self {
            id: r.id.clone(),
            model: r.model.clone(),
            content,
            finish_reason,
            prompt_tokens: pt,
            completion_tokens: ct,
            total_tokens: tt,
            tool_calls,
            raw_json: serde_json::to_string(&r).unwrap_or_default(),
        }
    }
}

/// The main SwiftLLM gateway client.
///
/// Example
/// -------
/// ```python
/// from swiftllm import SwiftLLM
///
/// llm = SwiftLLM()
/// llm.add_provider("openai", api_key="sk-...")
/// resp = llm.completion("gpt-4o-mini", "What is the meaning of life?")
/// print(resp.content)
/// ```
#[pyclass(name = "SwiftLLM")]
pub struct PySwiftLLM {
    providers: HashMap<String, ProviderConfig>,
    cache_enabled: bool,
    cache_max_size: usize,
    cache_ttl_seconds: u64,
    rate_limit_enabled: bool,
    rate_limit_max_requests: u64,
    rate_limit_window_seconds: u64,
    auth_api_keys: Vec<String>,
    default_provider: Option<String>,
    runtime: Runtime,
    /// Lazily initialised on first call.
    state: Option<Arc<AppState>>,
}

#[pymethods]
impl PySwiftLLM {
    /// Create a new SwiftLLM instance.
    ///
    /// Parameters
    /// ----------
    /// cache : bool
    ///     Enable response caching (default True).
    /// cache_max_size : int
    ///     Maximum number of cached responses (default 1000).
    /// cache_ttl : int
    ///     Cache time-to-live in seconds (default 300).
    #[new]
    #[pyo3(signature = (*, cache=true, cache_max_size=1000, cache_ttl=300))]
    fn new(cache: bool, cache_max_size: usize, cache_ttl: u64) -> PyResult<Self> {
        let runtime = Runtime::new().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create Tokio runtime: {}", e))
        })?;
        Ok(Self {
            providers: HashMap::new(),
            cache_enabled: cache,
            cache_max_size,
            cache_ttl_seconds: cache_ttl,
            rate_limit_enabled: false,
            rate_limit_max_requests: 100,
            rate_limit_window_seconds: 60,
            auth_api_keys: Vec::new(),
            default_provider: None,
            runtime,
            state: None,
        })
    }

    /// Register a provider.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Provider identifier (e.g. "openai", "anthropic", "gemini", "mistral",
    ///     "ollama", "groq", "together", "bedrock").
    /// api_key : str, optional
    ///     The provider's API key. Not needed for Ollama or Bedrock.
    /// base_url : str, optional
    ///     Custom base URL for the provider API.
    /// models : list[str], optional
    ///     Explicit list of model names supported by this provider.
    /// priority : int
    ///     Failover priority (lower = tried first). Default 100.
    #[pyo3(signature = (name, *, api_key=None, base_url=None, models=None, priority=100))]
    fn add_provider(
        &mut self,
        name: &str,
        api_key: Option<String>,
        base_url: Option<String>,
        models: Option<Vec<String>>,
        priority: u32,
    ) -> PyResult<()> {
        let kind = parse_kind(name)?;
        self.providers.insert(
            name.to_lowercase(),
            ProviderConfig {
                kind,
                api_key: api_key.map(SecretString::from),
                base_url,
                models: models.unwrap_or_default(),
                priority,
            },
        );
        // Invalidate cached state so it rebuilds on next call
        self.state = None;
        Ok(())
    }

    /// Set an API key for a provider using environment-variable style.
    /// Convenience wrapper: `llm.set_key("openai", "sk-...")`
    fn set_key(&mut self, provider: &str, api_key: &str) -> PyResult<()> {
        let name = provider.to_lowercase();
        if let Some(cfg) = self.providers.get_mut(&name) {
            cfg.api_key = Some(SecretString::from(api_key.to_string()));
            self.state = None;
            Ok(())
        } else {
            // Auto-register with defaults
            self.add_provider(&name, Some(api_key.to_string()), None, None, 100)
        }
    }

    /// Send a chat completion request.
    ///
    /// Parameters
    /// ----------
    /// model : str
    ///     Model identifier (e.g. "gpt-4o-mini", "claude-sonnet-4-20250514").
    /// messages : list[dict] | str
    ///     Either a list of ``{"role": ..., "content": ...}`` dicts,
    ///     or a plain string (which becomes a single user message).
    /// temperature : float, optional
    /// max_tokens : int, optional
    /// top_p : float, optional
    /// tools : list[dict], optional
    ///     Tool definitions for function calling.
    /// tool_choice : str | dict, optional
    ///     Controls tool usage: "auto", "none", "required", or a dict.
    ///
    /// Returns
    /// -------
    /// ChatCompletionResponse
    #[pyo3(signature = (model, messages, *, temperature=None, max_tokens=None, top_p=None, tools=None, tool_choice=None))]
    fn completion(
        &mut self,
        py: Python<'_>,
        model: &str,
        messages: &Bound<'_, pyo3::types::PyAny>,
        temperature: Option<f64>,
        max_tokens: Option<u64>,
        top_p: Option<f64>,
        tools: Option<&Bound<'_, pyo3::types::PyAny>>,
        tool_choice: Option<&Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<PyChatResponse> {
        let msgs = extract_messages(messages)?;
        let tools_vec = extract_tools(tools)?;
        let tool_choice_val = extract_tool_choice(tool_choice)?;
        let state = self.ensure_state()?;

        let request = ChatRequest {
            model: model.to_string(),
            messages: msgs,
            temperature,
            max_tokens,
            top_p,
            stream: Some(false),
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            tools: tools_vec,
            tool_choice: tool_choice_val,
        };

        // Find the right provider
        let (provider_name, _) = state
            .config
            .find_provider_for_model(&request.model)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "No provider configured for model '{}'. Add one with add_provider() first.",
                    request.model
                ))
            })?;

        let provider: Arc<dyn Provider> = state
            .providers
            .get(provider_name)
            .cloned()
            .ok_or_else(|| PyRuntimeError::new_err("Provider not initialised"))?;

        // Release the GIL while we do the network call
        let response = py.allow_threads(|| {
            self.runtime
                .block_on(async { provider.chat(&request).await })
        });

        let chat_resp = response.map_err(provider_err_to_py)?;

        Ok(PyChatResponse::from(chat_resp))
    }

    /// List all currently configured providers.
    fn list_providers(&self) -> Vec<String> {
        self.providers.keys().cloned().collect()
    }

    /// List all models across all configured providers.
    fn list_models(&mut self) -> PyResult<Vec<HashMap<String, String>>> {
        let state = self.ensure_state()?;
        let mut out = Vec::new();
        for (name, cfg) in &state.config.providers {
            for model in &cfg.models {
                let mut m = HashMap::new();
                m.insert("id".into(), model.clone());
                m.insert("provider".into(), name.clone());
                out.push(m);
            }
        }
        Ok(out)
    }

    /// Start the SwiftLLM gateway HTTP server (blocking).
    ///
    /// This lets you run `swiftllm.serve(port=8080)` from Python
    /// as an alternative to the CLI binary.
    #[pyo3(signature = (*, port=8080))]
    fn serve(&mut self, py: Python<'_>, port: u16) -> PyResult<()> {
        let state = self.ensure_state()?;
        let app = crate::server::build_router(state);

        let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
        println!("swiftllm serving on http://{}", addr);

        py.allow_threads(|| {
            self.runtime.block_on(async {
                let listener = tokio::net::TcpListener::bind(addr)
                    .await
                    .map_err(|e| PyRuntimeError::new_err(format!("Bind failed: {}", e)))?;
                axum::serve(listener, app)
                    .await
                    .map_err(|e| PyRuntimeError::new_err(format!("Server error: {}", e)))
            })
        })
    }

    fn __repr__(&self) -> String {
        let names: Vec<&str> = self.providers.keys().map(|s| s.as_str()).collect();
        format!("SwiftLLM(providers={:?})", names)
    }
}

impl PySwiftLLM {
    /// Build (or return cached) AppState.
    fn ensure_state(&mut self) -> PyResult<Arc<AppState>> {
        if let Some(ref s) = self.state {
            return Ok(s.clone());
        }

        if self.providers.is_empty() {
            return Err(PyValueError::new_err(
                "No providers configured. Call add_provider() first.",
            ));
        }

        let config = Config {
            port: 0, // unused in library mode
            auth: AuthConfig {
                api_keys: self
                    .auth_api_keys
                    .iter()
                    .map(|s| SecretString::from(s.clone()))
                    .collect(),
            },
            providers: self.providers.clone(),
            routing: RoutingConfig {
                default_provider: self.default_provider.clone(),
            },
            cache: CacheConfig {
                enabled: self.cache_enabled,
                max_size: self.cache_max_size,
                ttl_seconds: self.cache_ttl_seconds,
            },
            rate_limit: RateLimitConfig {
                enabled: self.rate_limit_enabled,
                max_requests: self.rate_limit_max_requests,
                window_seconds: self.rate_limit_window_seconds,
                providers: HashMap::new(),
            },
        };

        let state = Arc::new(AppState::new(config));
        self.state = Some(state.clone());
        Ok(state)
    }
}

// ---------------------------------------------------------------------------
// Convenience free-function (LiteLLM-style one-liner)
// ---------------------------------------------------------------------------

/// Quick one-shot completion — the LiteLLM-style API.
///
/// ```python
/// from swiftllm import completion
///
/// resp = completion(
///     model="gpt-4o-mini",
///     messages=[{"role": "user", "content": "Hello!"}],
///     api_key="sk-...",
/// )
/// print(resp.content)
/// ```
#[pyfunction]
#[pyo3(signature = (model, messages, *, api_key=None, base_url=None, temperature=None, max_tokens=None, top_p=None, tools=None, tool_choice=None))]
fn completion(
    py: Python<'_>,
    model: &str,
    messages: &Bound<'_, pyo3::types::PyAny>,
    api_key: Option<String>,
    base_url: Option<String>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    top_p: Option<f64>,
    tools: Option<&Bound<'_, pyo3::types::PyAny>>,
    tool_choice: Option<&Bound<'_, pyo3::types::PyAny>>,
) -> PyResult<PyChatResponse> {
    // Infer provider from model name
    let provider_name = infer_provider(model)?;

    let mut llm = PySwiftLLM::new(true, 1000, 300)?;
    llm.add_provider(&provider_name, api_key, base_url, None, 100)?;
    llm.completion(
        py,
        model,
        messages,
        temperature,
        max_tokens,
        top_p,
        tools,
        tool_choice,
    )
}

fn infer_provider(model: &str) -> PyResult<String> {
    let m = model.to_lowercase();
    if m.starts_with("gpt-")
        || m.starts_with("o1")
        || m.starts_with("o3")
        || m.starts_with("o4")
        || m.starts_with("chatgpt-")
    {
        Ok("openai".into())
    } else if m.starts_with("claude-") {
        Ok("anthropic".into())
    } else if m.starts_with("gemini-") {
        Ok("gemini".into())
    } else if m.starts_with("mistral-")
        || m.starts_with("codestral")
        || m.starts_with("pixtral")
        || m.starts_with("ministral")
    {
        Ok("mistral".into())
    } else if m.starts_with("llama") || m.starts_with("mixtral") || m.starts_with("gemma") {
        Ok("groq".into())
    } else if m.contains('/') {
        Ok("together".into())
    } else if m.starts_with("anthropic.")
        || m.starts_with("amazon.")
        || m.starts_with("meta.")
        || m.starts_with("cohere.")
    {
        Ok("bedrock".into())
    } else if m.contains(':') {
        Ok("ollama".into())
    } else {
        Err(PyValueError::new_err(format!(
            "Cannot infer provider for model '{}'. Pass provider explicitly via SwiftLLM class.",
            model
        )))
    }
}

// ---------------------------------------------------------------------------
// Message extraction helpers
// ---------------------------------------------------------------------------

fn extract_messages(obj: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Vec<Message>> {
    // Case 1: plain string → single user message
    if let Ok(text) = obj.extract::<String>() {
        return Ok(vec![Message {
            role: "user".to_string(),
            content: Some(text),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }]);
    }

    // Case 2: list of dicts / PyMessage objects
    let list: Vec<Bound<'_, pyo3::types::PyAny>> = obj.extract()?;
    let mut msgs = Vec::with_capacity(list.len());
    for item in &list {
        // Try as PyMessage first
        if let Ok(pm) = item.extract::<PyMessage>() {
            msgs.push(Message {
                role: pm.role,
                content: pm.content,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
            continue;
        }
        // Otherwise treat as dict
        let dict: &Bound<'_, pyo3::types::PyDict> = item.downcast()?;
        let role: String = dict
            .get_item("role")?
            .ok_or_else(|| PyValueError::new_err("Message dict must have 'role' key"))?
            .extract()?;
        let content: Option<String> = dict.get_item("content")?.map(|v| v.extract()).transpose()?;
        let tool_call_id: Option<String> = dict
            .get_item("tool_call_id")?
            .map(|v| v.extract())
            .transpose()?;
        let name: Option<String> = dict.get_item("name")?.map(|v| v.extract()).transpose()?;

        // Parse tool_calls if present
        let tool_calls = if let Some(tc_list) = dict.get_item("tool_calls")? {
            let json_str = py_to_json_value(&tc_list)?;
            serde_json::from_value(json_str).ok()
        } else {
            None
        };

        msgs.push(Message {
            role,
            content,
            tool_calls,
            tool_call_id,
            name,
        });
    }
    Ok(msgs)
}

fn extract_tools(
    tools: Option<&Bound<'_, pyo3::types::PyAny>>,
) -> PyResult<Option<Vec<ToolDefinition>>> {
    match tools {
        None => Ok(None),
        Some(obj) => {
            let json_val = py_to_json_value(obj)?;
            let tools: Vec<ToolDefinition> = serde_json::from_value(json_val)
                .map_err(|e| PyValueError::new_err(format!("Invalid tools format: {}", e)))?;
            Ok(Some(tools))
        }
    }
}

fn extract_tool_choice(
    tool_choice: Option<&Bound<'_, pyo3::types::PyAny>>,
) -> PyResult<Option<serde_json::Value>> {
    match tool_choice {
        None => Ok(None),
        Some(obj) => {
            let val = py_to_json_value(obj)?;
            Ok(Some(val))
        }
    }
}

/// Convert a Python object to serde_json::Value via JSON serialization.
fn py_to_json_value(obj: &Bound<'_, pyo3::types::PyAny>) -> PyResult<serde_json::Value> {
    let py = obj.py();
    let json_module = py.import_bound("json")?;
    let json_str: String = json_module.call_method1("dumps", (obj,))?.extract()?;
    serde_json::from_str(&json_str)
        .map_err(|e| PyValueError::new_err(format!("JSON parse error: {}", e)))
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// The native Python module. maturin will look for this function.
#[pymodule]
pub fn _swiftllm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySwiftLLM>()?;
    m.add_class::<PyMessage>()?;
    m.add_class::<PyChatResponse>()?;
    m.add_class::<PyToolCall>()?;
    m.add_function(wrap_pyfunction!(completion, m)?)?;
    Ok(())
}
