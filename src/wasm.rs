//! WebAssembly bindings for SwiftLLM.
//!
//! Provides a JavaScript-friendly API for running SwiftLLM in the browser
//! or any WASM runtime. Uses `reqwest` with the `web-sys` backend for
//! HTTP requests (the browser's `fetch` API under the hood).
//!
//! # Build
//!
//! ```sh
//! wasm-pack build --target web --features wasm
//! ```

use std::collections::HashMap;

use wasm_bindgen::prelude::*;

// ── Internal state ─────────────────────────────────────────────────────────

struct WasmProvider {
    name: String,
    api_key: String,
    base_url: Option<String>,
}

/// A SwiftLLM instance for use in WASM environments.
///
/// Unlike the FFI handle, this does not create a Tokio runtime — WASM uses
/// the browser's event loop via `wasm-bindgen-futures`.
#[wasm_bindgen]
pub struct WasmSwiftLLM {
    providers: HashMap<String, WasmProvider>,
}

#[wasm_bindgen]
impl WasmSwiftLLM {
    /// Create a new SwiftLLM WASM instance.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    /// Register a provider with the given name and API key.
    #[wasm_bindgen]
    pub fn add_provider(
        &mut self,
        name: &str,
        api_key: &str,
        base_url: Option<String>,
    ) -> Result<(), JsValue> {
        self.providers.insert(
            name.to_lowercase(),
            WasmProvider {
                name: name.to_lowercase(),
                api_key: api_key.to_string(),
                base_url,
            },
        );
        Ok(())
    }

    /// Send a chat completion request and return the JSON response.
    #[wasm_bindgen]
    pub async fn completion(&self, model: &str, prompt: &str) -> Result<JsValue, JsValue> {
        let provider_name = infer_provider(model)
            .ok_or_else(|| JsValue::from_str("Cannot infer provider for model"))?;

        let provider = self.providers.get(&provider_name).ok_or_else(|| {
            JsValue::from_str(&format!("Provider '{}' not configured", provider_name))
        })?;

        let base_url = provider
            .base_url
            .as_deref()
            .unwrap_or(default_base_url(&provider_name));

        let body = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        });

        let client = reqwest::Client::new();
        let resp = client
            .post(format!(
                "{}/chat/completions",
                base_url.trim_end_matches('/')
            ))
            .header("Authorization", format!("Bearer {}", provider.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| JsValue::from_str(&format!("Network error: {}", e)))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(JsValue::from_str(&format!("API error: {}", text)));
        }

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| JsValue::from_str(&format!("Parse error: {}", e)))?;

        serde_wasm_bindgen::to_value(&json).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl Default for WasmSwiftLLM {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot completion: infer provider from model name.
#[wasm_bindgen]
pub async fn quick_completion(
    model: &str,
    prompt: &str,
    api_key: &str,
) -> Result<JsValue, JsValue> {
    let mut instance = WasmSwiftLLM::new();
    let provider_name = infer_provider(model)
        .ok_or_else(|| JsValue::from_str("Cannot infer provider for model"))?;
    instance.add_provider(&provider_name, api_key, None)?;
    instance.completion(model, prompt).await
}

// ── Helpers ────────────────────────────────────────────────────────────────

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
    } else {
        None
    }
}

fn default_base_url(provider: &str) -> &str {
    match provider {
        "openai" => "https://api.openai.com/v1",
        "anthropic" => "https://api.anthropic.com/v1",
        "gemini" => "https://generativelanguage.googleapis.com/v1beta",
        "mistral" => "https://api.mistral.ai/v1",
        "groq" => "https://api.groq.com/openai/v1",
        "together" => "https://api.together.xyz/v1",
        _ => "https://api.openai.com/v1",
    }
}
