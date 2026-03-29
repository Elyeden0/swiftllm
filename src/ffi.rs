//! C FFI bindings for SwiftLLM.
//!
//! This module provides a C-compatible API so SwiftLLM can be called from
//! any language with C FFI support: C, C++, Go, Java (JNI/Panama), Ruby,
//! PHP, C#, Elixir, Swift, Kotlin, Zig, etc.
//!
//! # Memory Safety
//!
//! All strings returned by these functions are heap-allocated and must be
//! freed by calling `swiftllm_free_string`. Failing to do so will leak memory.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::Arc;

use tokio::runtime::Runtime;

use crate::config::{
    AuthConfig, CacheConfig, Config, ProviderConfig, ProviderKind, RateLimitConfig, RoutingConfig,
};
use crate::providers::types::{ChatRequest, Message};
use crate::providers::Provider;
use crate::server::AppState;

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque handle to a SwiftLLM instance.
pub struct SwiftLLMHandle {
    providers: HashMap<String, ProviderConfig>,
    runtime: Runtime,
    state: Option<Arc<AppState>>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a C string pointer to a Rust &str. Returns None if null or invalid UTF-8.
unsafe fn cstr_to_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    CStr::from_ptr(ptr).to_str().ok()
}

/// Allocate a C string on the heap. Caller must free with `swiftllm_free_string`.
fn to_c_string(s: &str) -> *mut c_char {
    CString::new(s).unwrap_or_default().into_raw()
}

fn parse_kind(s: &str) -> Option<ProviderKind> {
    match s.to_lowercase().as_str() {
        "openai" => Some(ProviderKind::Openai),
        "anthropic" => Some(ProviderKind::Anthropic),
        "gemini" => Some(ProviderKind::Gemini),
        "mistral" => Some(ProviderKind::Mistral),
        "ollama" => Some(ProviderKind::Ollama),
        "groq" => Some(ProviderKind::Groq),
        "together" => Some(ProviderKind::Together),
        "bedrock" => Some(ProviderKind::Bedrock),
        _ => None,
    }
}

fn infer_provider(model: &str) -> Option<String> {
    let m = model.to_lowercase();
    if m.starts_with("gpt-")
        || m.starts_with("o1")
        || m.starts_with("o3")
        || m.starts_with("o4")
        || m.starts_with("chatgpt-")
    {
        Some("openai".into())
    } else if m.starts_with("claude-") {
        Some("anthropic".into())
    } else if m.starts_with("gemini-") {
        Some("gemini".into())
    } else if m.starts_with("mistral-")
        || m.starts_with("codestral")
        || m.starts_with("pixtral")
        || m.starts_with("ministral")
    {
        Some("mistral".into())
    } else if m.starts_with("llama") || m.starts_with("mixtral") || m.starts_with("gemma") {
        Some("groq".into())
    } else if m.contains('/') {
        Some("together".into())
    } else if m.starts_with("anthropic.")
        || m.starts_with("amazon.")
        || m.starts_with("meta.")
        || m.starts_with("cohere.")
    {
        Some("bedrock".into())
    } else if m.contains(':') {
        Some("ollama".into())
    } else {
        None
    }
}

impl SwiftLLMHandle {
    fn ensure_state(&mut self) -> Result<Arc<AppState>, String> {
        if let Some(ref s) = self.state {
            return Ok(s.clone());
        }
        if self.providers.is_empty() {
            return Err("No providers configured".into());
        }
        let config = Config {
            port: 0,
            auth: AuthConfig::default(),
            providers: self.providers.clone(),
            routing: RoutingConfig {
                default_provider: None,
            },
            cache: CacheConfig::default(),
            rate_limit: RateLimitConfig::default(),
        };
        let state = Arc::new(AppState::new(config));
        self.state = Some(state.clone());
        Ok(state)
    }
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

/// Create a new SwiftLLM instance. Returns an opaque pointer.
/// Must be freed with `swiftllm_destroy`.
#[no_mangle]
pub extern "C" fn swiftllm_create() -> *mut SwiftLLMHandle {
    let runtime = match Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return ptr::null_mut(),
    };
    let handle = Box::new(SwiftLLMHandle {
        providers: HashMap::new(),
        runtime,
        state: None,
    });
    Box::into_raw(handle)
}

/// Destroy a SwiftLLM instance and free all resources.
///
/// # Safety
/// `handle` must be a valid pointer returned by `swiftllm_create`, or null.
#[no_mangle]
pub unsafe extern "C" fn swiftllm_destroy(handle: *mut SwiftLLMHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Register a provider with the given name and API key.
///
/// - `name`: Provider identifier (e.g. "openai", "anthropic", "gemini").
/// - `api_key`: The provider's API key (may be null for Ollama).
/// - `base_url`: Custom base URL (may be null for defaults).
///
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// `handle` must be a valid pointer. String pointers must be valid C strings or null.
#[no_mangle]
pub unsafe extern "C" fn swiftllm_add_provider(
    handle: *mut SwiftLLMHandle,
    name: *const c_char,
    api_key: *const c_char,
    base_url: *const c_char,
) -> i32 {
    let handle = match handle.as_mut() {
        Some(h) => h,
        None => return -1,
    };
    let name_str = match cstr_to_str(name) {
        Some(s) => s,
        None => return -1,
    };
    let kind = match parse_kind(name_str) {
        Some(k) => k,
        None => return -1,
    };
    let api_key_str = cstr_to_str(api_key).map(|s| {
        use secrecy::SecretString;
        SecretString::from(s.to_string())
    });
    let base_url_str = cstr_to_str(base_url).map(|s| s.to_string());

    handle.providers.insert(
        name_str.to_lowercase(),
        ProviderConfig {
            kind,
            api_key: api_key_str,
            base_url: base_url_str,
            models: Vec::new(),
            priority: 100,
        },
    );
    handle.state = None; // invalidate
    0
}

/// Send a chat completion request.
///
/// - `model`: Model name (e.g. "gpt-4o-mini").
/// - `prompt`: User message text.
///
/// Returns a heap-allocated JSON string with the response, or null on error.
/// The caller must free the returned string with `swiftllm_free_string`.
///
/// # Safety
/// `handle` must be a valid pointer. String pointers must be valid C strings.
#[no_mangle]
pub unsafe extern "C" fn swiftllm_completion(
    handle: *mut SwiftLLMHandle,
    model: *const c_char,
    prompt: *const c_char,
) -> *mut c_char {
    let handle = match handle.as_mut() {
        Some(h) => h,
        None => return ptr::null_mut(),
    };
    let model_str = match cstr_to_str(model) {
        Some(s) => s,
        None => return ptr::null_mut(),
    };
    let prompt_str = match cstr_to_str(prompt) {
        Some(s) => s,
        None => return ptr::null_mut(),
    };

    let state = match handle.ensure_state() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let request = ChatRequest {
        model: model_str.to_string(),
        messages: vec![Message {
            role: "user".to_string(),
            content: Some(prompt_str.to_string()),
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
    };

    let provider_name = state
        .config
        .find_provider_for_model(&request.model)
        .map(|(n, _)| n.clone());

    let provider_name = match provider_name {
        Some(n) => n,
        None => return ptr::null_mut(),
    };

    let provider: Arc<dyn Provider> = match state.providers.get(&provider_name) {
        Some(p) => p.clone(),
        None => return ptr::null_mut(),
    };

    let result = handle
        .runtime
        .block_on(async { provider.chat(&request).await });

    match result {
        Ok(resp) => {
            let json = serde_json::to_string(&resp).unwrap_or_default();
            to_c_string(&json)
        }
        Err(_) => ptr::null_mut(),
    }
}

/// One-shot completion: infer provider from model name.
///
/// - `model`: Model name (e.g. "gpt-4o-mini", "claude-sonnet-4-6").
/// - `prompt`: User message text.
/// - `api_key`: API key for the inferred provider.
///
/// Returns a heap-allocated JSON string with the response, or null on error.
/// The caller must free the returned string with `swiftllm_free_string`.
///
/// # Safety
/// All string pointers must be valid C strings.
#[no_mangle]
pub unsafe extern "C" fn swiftllm_quick_completion(
    model: *const c_char,
    prompt: *const c_char,
    api_key: *const c_char,
) -> *mut c_char {
    let model_str = match cstr_to_str(model) {
        Some(s) => s,
        None => return ptr::null_mut(),
    };

    let handle = swiftllm_create();
    if handle.is_null() {
        return ptr::null_mut();
    }

    let provider_name = match infer_provider(model_str) {
        Some(p) => p,
        None => {
            swiftllm_destroy(handle);
            return ptr::null_mut();
        }
    };

    let provider_c = CString::new(provider_name).unwrap_or_default();
    if swiftllm_add_provider(handle, provider_c.as_ptr(), api_key, ptr::null()) != 0 {
        swiftllm_destroy(handle);
        return ptr::null_mut();
    }

    let result = swiftllm_completion(handle, model, prompt);
    swiftllm_destroy(handle);
    result
}

/// Free a string previously returned by swiftllm functions.
///
/// # Safety
/// `ptr` must be a pointer returned by a swiftllm function, or null.
#[no_mangle]
pub unsafe extern "C" fn swiftllm_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}

/// Return the library version as a static string. Do NOT free this pointer.
#[no_mangle]
pub extern "C" fn swiftllm_version() -> *const c_char {
    // Include a null terminator
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}
