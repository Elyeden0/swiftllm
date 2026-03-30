//! Integration tests for SwiftLLM — cross-cutting tests for config routing,
//! cache behaviour, rate limiting, and failover logic.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use swiftllm::config::{
    CacheConfig, Config, ProviderConfig, ProviderKind, RateLimitConfig, RoutingConfig,
};
use swiftllm::failover::chat_with_failover;
use swiftllm::middleware::cache::ResponseCache;
use swiftllm::middleware::rate_limit::{RateLimitConfig as RlConfig, RateLimiter};
use swiftllm::providers::types::{ChatRequest, ChatResponse, Message, Usage};
use swiftllm::providers::{Provider, ProviderError};

use async_trait::async_trait;
use futures::stream::BoxStream;
use secrecy::SecretString;
use swiftllm::providers::types::StreamChunk;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn make_message(role: &str, content: &str) -> Message {
    Message {
        role: role.to_string(),
        content: Some(content.to_string()),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }
}

fn make_request(model: &str) -> ChatRequest {
    ChatRequest {
        model: model.to_string(),
        messages: vec![make_message("user", "Hello")],
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

fn make_config(providers: Vec<(&str, ProviderKind, Vec<&str>, u32)>) -> Config {
    let mut provider_map = HashMap::new();
    for (name, kind, models, priority) in providers {
        provider_map.insert(
            name.to_string(),
            ProviderConfig {
                kind,
                api_key: Some(SecretString::from("test-key".to_string())),
                base_url: None,
                models: models.into_iter().map(String::from).collect(),
                priority,
            },
        );
    }
    Config {
        port: 8080,
        auth: Default::default(),
        providers: provider_map,
        routing: RoutingConfig {
            default_provider: None,
        },
        cache: CacheConfig::default(),
        rate_limit: swiftllm::config::RateLimitConfig::default(),
    }
}

/// A mock provider that always succeeds.
struct MockProvider {
    name: String,
    response_content: String,
}

#[async_trait]
impl Provider for MockProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        Ok(ChatResponse::new(
            request.model.clone(),
            self.response_content.clone(),
            Some(Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            }),
        ))
    }

    async fn chat_stream(
        &self,
        _request: &ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, ProviderError>>, ProviderError> {
        Err(ProviderError::Config(
            "streaming not implemented in mock".to_string(),
        ))
    }
}

/// A mock provider that always fails.
struct FailingProvider {
    name: String,
}

#[async_trait]
impl Provider for FailingProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn chat(&self, _request: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        Err(ProviderError::Api {
            status: 500,
            message: "mock failure".to_string(),
        })
    }

    async fn chat_stream(
        &self,
        _request: &ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, ProviderError>>, ProviderError> {
        Err(ProviderError::Api {
            status: 500,
            message: "mock failure".to_string(),
        })
    }
}

// ── Config routing tests ────────────────────────────────────────────────────

#[test]
fn test_exact_model_match() {
    let config = make_config(vec![
        ("openai", ProviderKind::Openai, vec!["gpt-4o", "gpt-4"], 100),
        (
            "anthropic",
            ProviderKind::Anthropic,
            vec!["claude-3-opus"],
            100,
        ),
    ]);

    let (name, _) = config.find_provider_for_model("claude-3-opus").unwrap();
    assert_eq!(name, "anthropic");

    let (name, _) = config.find_provider_for_model("gpt-4o").unwrap();
    assert_eq!(name, "openai");
}

#[test]
fn test_prefix_match() {
    let config = make_config(vec![
        ("openai", ProviderKind::Openai, vec![], 100),
        ("anthropic", ProviderKind::Anthropic, vec![], 100),
        ("gemini", ProviderKind::Gemini, vec![], 100),
    ]);

    let (name, _) = config.find_provider_for_model("gpt-4-turbo").unwrap();
    assert_eq!(name, "openai");

    let (name, _) = config.find_provider_for_model("claude-3-sonnet").unwrap();
    assert_eq!(name, "anthropic");

    let (name, _) = config.find_provider_for_model("gemini-pro").unwrap();
    assert_eq!(name, "gemini");
}

#[test]
fn test_default_provider_fallback() {
    let mut config = make_config(vec![("openai", ProviderKind::Openai, vec![], 100)]);
    config.routing.default_provider = Some("openai".to_string());

    // "unknown-model" doesn't match any prefix, should fall back to default
    let (name, _) = config.find_provider_for_model("unknown-model").unwrap();
    assert_eq!(name, "openai");
}

#[test]
fn test_no_match_returns_none() {
    let config = make_config(vec![("openai", ProviderKind::Openai, vec![], 100)]);

    // No default provider set, model doesn't match any prefix
    assert!(config.find_provider_for_model("unknown-model").is_none());
}

#[test]
fn test_providers_by_priority_sorting() {
    let config = make_config(vec![
        ("slow", ProviderKind::Openai, vec![], 300),
        ("fast", ProviderKind::Anthropic, vec![], 10),
        ("mid", ProviderKind::Gemini, vec![], 100),
    ]);

    let sorted = config.providers_by_priority();
    let names: Vec<&str> = sorted.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, vec!["fast", "mid", "slow"]);
}

// ── Cache tests ─────────────────────────────────────────────────────────────

#[test]
fn test_cache_hit_and_miss() {
    let cache = ResponseCache::new(100, 300);
    let req = make_request("gpt-4");

    // Miss initially
    assert!(cache.get(&req).is_none());

    let response = ChatResponse::new("gpt-4".to_string(), "cached answer".to_string(), None);
    cache.put(&req, response.clone());

    // Hit after put
    let cached = cache.get(&req).unwrap();
    assert_eq!(
        cached.choices[0].message.content,
        Some("cached answer".to_string())
    );
}

#[test]
fn test_cache_different_requests() {
    let cache = ResponseCache::new(100, 300);

    let req1 = make_request("gpt-4");
    let req2 = make_request("claude-3");

    let resp1 = ChatResponse::new("gpt-4".to_string(), "answer 1".to_string(), None);
    let resp2 = ChatResponse::new("claude-3".to_string(), "answer 2".to_string(), None);

    cache.put(&req1, resp1);
    cache.put(&req2, resp2);

    let cached1 = cache.get(&req1).unwrap();
    assert_eq!(
        cached1.choices[0].message.content,
        Some("answer 1".to_string())
    );

    let cached2 = cache.get(&req2).unwrap();
    assert_eq!(
        cached2.choices[0].message.content,
        Some("answer 2".to_string())
    );
}

#[test]
fn test_cache_eviction_at_max_size() {
    let cache = ResponseCache::new(2, 300); // max 2 entries

    let req1 = make_request("model-a");
    let req2 = make_request("model-b");
    let req3 = make_request("model-c");

    cache.put(
        &req1,
        ChatResponse::new("model-a".to_string(), "a".to_string(), None),
    );
    cache.put(
        &req2,
        ChatResponse::new("model-b".to_string(), "b".to_string(), None),
    );

    assert_eq!(cache.stats().size, 2);

    // Adding a third entry should evict the oldest (req1 hasn't been accessed)
    cache.put(
        &req3,
        ChatResponse::new("model-c".to_string(), "c".to_string(), None),
    );
    assert_eq!(cache.stats().size, 2);

    // req3 should be present
    assert!(cache.get(&req3).is_some());
}

#[test]
fn test_cache_stats() {
    let cache = ResponseCache::new(100, 300);
    assert_eq!(cache.stats().size, 0);
    assert_eq!(cache.stats().max_size, 100);

    cache.put(
        &make_request("m1"),
        ChatResponse::new("m1".to_string(), "x".to_string(), None),
    );
    assert_eq!(cache.stats().size, 1);
}

// ── Rate limiter tests ──────────────────────────────────────────────────────

#[test]
fn test_rate_limiter_allows_under_limit() {
    let mut limits = HashMap::new();
    limits.insert(
        "openai".to_string(),
        RlConfig {
            max_requests: 5,
            window: Duration::from_secs(60),
        },
    );

    let limiter = RateLimiter::new(limits, None);

    // First 5 requests should succeed
    for _ in 0..5 {
        assert!(limiter.check("openai").is_ok());
    }
}

#[test]
fn test_rate_limiter_rejects_when_exhausted() {
    let mut limits = HashMap::new();
    limits.insert(
        "openai".to_string(),
        RlConfig {
            max_requests: 2,
            window: Duration::from_secs(60),
        },
    );

    let limiter = RateLimiter::new(limits, None);

    assert!(limiter.check("openai").is_ok());
    assert!(limiter.check("openai").is_ok());
    // Third request should be rejected
    let result = limiter.check("openai");
    assert!(result.is_err());
    // retry_after should be positive
    assert!(result.unwrap_err() > 0.0);
}

#[test]
fn test_rate_limiter_no_limit_allows_all() {
    let limiter = RateLimiter::new(HashMap::new(), None);

    // No limits configured → should always allow
    for _ in 0..100 {
        assert!(limiter.check("any_provider").is_ok());
    }
}

#[test]
fn test_rate_limiter_default_limit() {
    let default = RlConfig {
        max_requests: 1,
        window: Duration::from_secs(60),
    };
    let limiter = RateLimiter::new(HashMap::new(), Some(default));

    assert!(limiter.check("unknown_provider").is_ok());
    assert!(limiter.check("unknown_provider").is_err());
}

#[test]
fn test_rate_limiter_per_provider_isolation() {
    let mut limits = HashMap::new();
    limits.insert(
        "openai".to_string(),
        RlConfig {
            max_requests: 1,
            window: Duration::from_secs(60),
        },
    );
    limits.insert(
        "anthropic".to_string(),
        RlConfig {
            max_requests: 1,
            window: Duration::from_secs(60),
        },
    );

    let limiter = RateLimiter::new(limits, None);

    // Exhaust openai
    assert!(limiter.check("openai").is_ok());
    assert!(limiter.check("openai").is_err());

    // Anthropic should still have its own bucket
    assert!(limiter.check("anthropic").is_ok());
}

// ── Failover tests ──────────────────────────────────────────────────────────

#[tokio::test]
async fn test_failover_succeeds_on_first_provider() {
    let providers: Vec<(&str, Arc<dyn Provider>)> = vec![(
        "primary",
        Arc::new(MockProvider {
            name: "primary".to_string(),
            response_content: "hello from primary".to_string(),
        }),
    )];

    let request = make_request("test-model");
    let (name, response) = chat_with_failover(&providers, &request).await.unwrap();

    assert_eq!(name, "primary");
    assert_eq!(
        response.choices[0].message.content,
        Some("hello from primary".to_string())
    );
}

#[tokio::test]
async fn test_failover_skips_failed_provider() {
    let providers: Vec<(&str, Arc<dyn Provider>)> = vec![
        (
            "failing",
            Arc::new(FailingProvider {
                name: "failing".to_string(),
            }),
        ),
        (
            "backup",
            Arc::new(MockProvider {
                name: "backup".to_string(),
                response_content: "hello from backup".to_string(),
            }),
        ),
    ];

    let request = make_request("test-model");
    let (name, response) = chat_with_failover(&providers, &request).await.unwrap();

    assert_eq!(name, "backup");
    assert_eq!(
        response.choices[0].message.content,
        Some("hello from backup".to_string())
    );
}

#[tokio::test]
async fn test_failover_all_fail() {
    let providers: Vec<(&str, Arc<dyn Provider>)> = vec![
        (
            "fail1",
            Arc::new(FailingProvider {
                name: "fail1".to_string(),
            }),
        ),
        (
            "fail2",
            Arc::new(FailingProvider {
                name: "fail2".to_string(),
            }),
        ),
    ];

    let request = make_request("test-model");
    let result = chat_with_failover(&providers, &request).await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_failover_empty_chain() {
    let providers: Vec<(&str, Arc<dyn Provider>)> = vec![];
    let request = make_request("test-model");
    let result = chat_with_failover(&providers, &request).await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_failover_skips_multiple_failures() {
    let providers: Vec<(&str, Arc<dyn Provider>)> = vec![
        (
            "fail1",
            Arc::new(FailingProvider {
                name: "fail1".to_string(),
            }),
        ),
        (
            "fail2",
            Arc::new(FailingProvider {
                name: "fail2".to_string(),
            }),
        ),
        (
            "success",
            Arc::new(MockProvider {
                name: "success".to_string(),
                response_content: "third time's the charm".to_string(),
            }),
        ),
    ];

    let request = make_request("test-model");
    let (name, response) = chat_with_failover(&providers, &request).await.unwrap();

    assert_eq!(name, "success");
    assert_eq!(
        response.choices[0].message.content,
        Some("third time's the charm".to_string())
    );
}
