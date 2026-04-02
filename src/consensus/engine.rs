use std::collections::HashMap;
use std::time::Instant;

use tracing::{info, warn};

use crate::providers::types::{
    ChatRequest, ChatResponse, ConsensusConfig, ConsensusMetadata, ConsensusStrategy, Message,
};
use crate::providers::ProviderError;
use crate::server::AppState;

/// Result from a single model query within a consensus run.
struct ModelResult {
    model: String,
    response: ChatResponse,
    latency_ms: u64,
}

/// Engine for multi-model consensus execution.
pub struct ConsensusEngine;

impl ConsensusEngine {
    /// Execute a consensus request: fan out to all models, then apply the strategy.
    pub async fn execute(
        state: &AppState,
        request: &ChatRequest,
        config: &ConsensusConfig,
    ) -> Result<ChatResponse, ProviderError> {
        // 1. Dispatch all models in parallel
        let results = Self::fan_out(state, request, config).await?;

        if results.is_empty() {
            return Err(ProviderError::Config(
                "All consensus models failed".to_string(),
            ));
        }

        // 2. Apply the chosen strategy
        match config.strategy {
            ConsensusStrategy::BestOf => Self::best_of(state, request, config, results).await,
            ConsensusStrategy::Majority => Self::majority(state, request, config, results).await,
            ConsensusStrategy::Merge => Self::merge(state, request, config, results).await,
        }
    }

    /// Fan out the request to all configured models in parallel, collecting successes.
    async fn fan_out(
        state: &AppState,
        request: &ChatRequest,
        config: &ConsensusConfig,
    ) -> Result<Vec<ModelResult>, ProviderError> {
        let mut handles = Vec::new();

        for model_name in &config.models {
            // Find the provider for this model
            let provider_entry =
                state
                    .config
                    .find_provider_for_model(model_name)
                    .and_then(|(pname, _)| {
                        state
                            .providers
                            .get(pname.as_str())
                            .map(|p| (pname.clone(), p.clone()))
                    });

            let (provider_name, provider) = match provider_entry {
                Some(entry) => entry,
                None => {
                    warn!(model = %model_name, "No provider found for consensus model, skipping");
                    continue;
                }
            };

            // Clone the request with the target model
            let mut model_request = request.clone();
            model_request.model = model_name.clone();
            // Strip consensus config from sub-requests to avoid recursion
            model_request.consensus = None;

            let model = model_name.clone();
            let pname = provider_name.clone();

            handles.push(tokio::spawn(async move {
                let start = Instant::now();
                let result = provider.chat(&model_request).await;
                let latency_ms = start.elapsed().as_millis() as u64;
                (model, pname, result, latency_ms)
            }));
        }

        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok((model, _provider_name, Ok(response), latency_ms)) => {
                    info!(model = %model, latency_ms, "Consensus model responded");
                    results.push(ModelResult {
                        model,
                        response,
                        latency_ms,
                    });
                }
                Ok((model, _provider_name, Err(e), _)) => {
                    warn!(model = %model, error = %e, "Consensus model failed");
                }
                Err(e) => {
                    warn!(error = %e, "Consensus task panicked");
                }
            }
        }

        Ok(results)
    }

    /// Extract the text content from the first choice of a response.
    fn extract_content(response: &ChatResponse) -> String {
        response
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .unwrap_or("")
            .to_string()
    }

    /// Build a latency map from the results.
    fn latency_map(results: &[ModelResult]) -> HashMap<String, u64> {
        results
            .iter()
            .map(|r| (r.model.clone(), r.latency_ms))
            .collect()
    }

    /// Resolve the judge model name from the config (defaults to first model in list).
    fn judge_model(config: &ConsensusConfig) -> String {
        config
            .judge
            .clone()
            .unwrap_or_else(|| config.models.first().cloned().unwrap_or_default())
    }

    /// Extract the original user prompt from the request messages.
    fn extract_user_prompt(request: &ChatRequest) -> String {
        request
            .messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .and_then(|m| m.content.as_deref())
            .unwrap_or("")
            .to_string()
    }

    /// Send a judge/synthesis prompt to a model and return the text response.
    async fn send_judge_prompt(
        state: &AppState,
        judge_model: &str,
        prompt: &str,
    ) -> Result<String, ProviderError> {
        let (provider_name, _) = state
            .config
            .find_provider_for_model(judge_model)
            .ok_or_else(|| {
                ProviderError::Config(format!(
                    "No provider found for judge model: {}",
                    judge_model
                ))
            })?;

        let provider = state.providers.get(provider_name.as_str()).ok_or_else(|| {
            ProviderError::Config(format!("Provider '{}' not initialized", provider_name))
        })?;

        let judge_request = ChatRequest {
            model: judge_model.to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: Some(prompt.to_string()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            temperature: Some(0.0),
            max_tokens: Some(64),
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

        let response = provider.chat(&judge_request).await?;
        Ok(Self::extract_content(&response))
    }

    // ── Strategies ──────────────────────────────────────────────────────────

    /// best_of: ask a judge model to pick the best response.
    async fn best_of(
        state: &AppState,
        request: &ChatRequest,
        config: &ConsensusConfig,
        results: Vec<ModelResult>,
    ) -> Result<ChatResponse, ProviderError> {
        let judge = Self::judge_model(config);
        let user_prompt = Self::extract_user_prompt(request);
        let latencies = Self::latency_map(&results);

        // Build the judge prompt
        let mut prompt = format!(
            "You are evaluating responses from multiple AI models. \
             Pick the best one. Respond with ONLY the number (1, 2, 3, etc.) \
             of the best response.\n\nOriginal question: {}\n",
            user_prompt
        );

        for (i, result) in results.iter().enumerate() {
            let content = Self::extract_content(&result.response);
            prompt.push_str(&format!(
                "\nResponse {} ({}):\n{}\n",
                i + 1,
                result.model,
                content
            ));
        }

        // Ask the judge
        let judge_answer = Self::send_judge_prompt(state, &judge, &prompt).await?;

        // Parse the selection — find the first digit
        let selected_idx = judge_answer
            .chars()
            .find(|c| c.is_ascii_digit())
            .and_then(|c| c.to_digit(10))
            .map(|n| (n as usize).saturating_sub(1))
            .unwrap_or(0);

        let winner_idx = selected_idx.min(results.len() - 1);
        let winner = &results[winner_idx];

        let mut response = winner.response.clone();
        response.consensus_metadata = Some(ConsensusMetadata {
            strategy: "best_of".to_string(),
            models_queried: results.iter().map(|r| r.model.clone()).collect(),
            judge: Some(judge.clone()),
            winner: Some(winner.model.clone()),
            individual_latencies_ms: latencies,
        });

        Ok(response)
    }

    /// majority: pick the most common response, falling back to best_of.
    async fn majority(
        state: &AppState,
        request: &ChatRequest,
        config: &ConsensusConfig,
        results: Vec<ModelResult>,
    ) -> Result<ChatResponse, ProviderError> {
        let latencies = Self::latency_map(&results);

        // Count identical responses by trimmed content
        let mut freq: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, result) in results.iter().enumerate() {
            let content = Self::extract_content(&result.response).trim().to_string();
            freq.entry(content).or_default().push(i);
        }

        // Find the group with the most occurrences
        if let Some((_, indices)) = freq.iter().max_by_key(|(_, v)| v.len()) {
            if indices.len() > 1 {
                // True majority found
                let winner_idx = indices[0];
                let winner = &results[winner_idx];
                let mut response = winner.response.clone();
                response.consensus_metadata = Some(ConsensusMetadata {
                    strategy: "majority".to_string(),
                    models_queried: results.iter().map(|r| r.model.clone()).collect(),
                    judge: None,
                    winner: Some(winner.model.clone()),
                    individual_latencies_ms: latencies,
                });
                return Ok(response);
            }
        }

        // All unique — fall back to best_of
        info!("No majority found, falling back to best_of strategy");
        Self::best_of(state, request, config, results).await
    }

    /// merge: synthesize the best parts of all responses into one.
    async fn merge(
        state: &AppState,
        request: &ChatRequest,
        config: &ConsensusConfig,
        results: Vec<ModelResult>,
    ) -> Result<ChatResponse, ProviderError> {
        let judge = Self::judge_model(config);
        let user_prompt = Self::extract_user_prompt(request);
        let latencies = Self::latency_map(&results);
        let models_queried: Vec<String> = results.iter().map(|r| r.model.clone()).collect();

        // Build the synthesis prompt
        let mut prompt = format!(
            "You are synthesizing the best parts of multiple AI responses \
             into one comprehensive answer. Combine insights while eliminating \
             redundancy.\n\nOriginal question: {}\n",
            user_prompt
        );

        for (i, result) in results.iter().enumerate() {
            let content = Self::extract_content(&result.response);
            prompt.push_str(&format!(
                "\nResponse {} ({}):\n{}\n",
                i + 1,
                result.model,
                content
            ));
        }

        prompt.push_str("\nSynthesize these into the best possible answer:");

        // Send the synthesis request with a higher token limit
        let (provider_name, _) = state
            .config
            .find_provider_for_model(&judge)
            .ok_or_else(|| {
                ProviderError::Config(format!(
                    "No provider found for synthesizer model: {}",
                    judge
                ))
            })?;

        let provider = state.providers.get(provider_name.as_str()).ok_or_else(|| {
            ProviderError::Config(format!("Provider '{}' not initialized", provider_name))
        })?;

        let synth_request = ChatRequest {
            model: judge.clone(),
            messages: vec![Message {
                role: "user".to_string(),
                content: Some(prompt),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            temperature: Some(0.3),
            max_tokens: request.max_tokens,
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

        let mut response = provider.chat(&synth_request).await?;
        response.consensus_metadata = Some(ConsensusMetadata {
            strategy: "merge".to_string(),
            models_queried,
            judge: Some(judge),
            winner: None,
            individual_latencies_ms: latencies,
        });

        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::types::{ChatResponse, ConsensusConfig, ConsensusStrategy, Message};

    #[test]
    fn test_extract_content() {
        let response = ChatResponse::new("test".to_string(), "hello world".to_string(), None);
        assert_eq!(ConsensusEngine::extract_content(&response), "hello world");
    }

    #[test]
    fn test_extract_content_empty() {
        let response = ChatResponse {
            id: "test".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![],
            usage: None,
            consensus_metadata: None,
            routing_metadata: None,
        };
        assert_eq!(ConsensusEngine::extract_content(&response), "");
    }

    #[test]
    fn test_judge_model_default() {
        let config = ConsensusConfig {
            models: vec!["gpt-4o".to_string(), "claude-sonnet-4-6".to_string()],
            strategy: ConsensusStrategy::BestOf,
            judge: None,
        };
        assert_eq!(ConsensusEngine::judge_model(&config), "gpt-4o");
    }

    #[test]
    fn test_judge_model_explicit() {
        let config = ConsensusConfig {
            models: vec!["gpt-4o".to_string()],
            strategy: ConsensusStrategy::BestOf,
            judge: Some("claude-sonnet-4-6".to_string()),
        };
        assert_eq!(ConsensusEngine::judge_model(&config), "claude-sonnet-4-6");
    }

    #[test]
    fn test_latency_map() {
        let results = vec![
            ModelResult {
                model: "a".to_string(),
                response: ChatResponse::new("a".to_string(), "hi".to_string(), None),
                latency_ms: 100,
            },
            ModelResult {
                model: "b".to_string(),
                response: ChatResponse::new("b".to_string(), "hi".to_string(), None),
                latency_ms: 200,
            },
        ];
        let map = ConsensusEngine::latency_map(&results);
        assert_eq!(map.get("a"), Some(&100));
        assert_eq!(map.get("b"), Some(&200));
    }

    #[test]
    fn test_extract_user_prompt() {
        let request = ChatRequest {
            model: "test".to_string(),
            messages: vec![
                Message {
                    role: "system".to_string(),
                    content: Some("You are helpful.".to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
                Message {
                    role: "user".to_string(),
                    content: Some("What is 2+2?".to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
            ],
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
            consensus: None,
            routing: None,
        };
        assert_eq!(
            ConsensusEngine::extract_user_prompt(&request),
            "What is 2+2?"
        );
    }

    #[test]
    fn test_consensus_config_deserialize() {
        let json = r#"{
            "models": ["gpt-4o", "claude-sonnet-4-6"],
            "strategy": "best_of",
            "judge": "claude-sonnet-4-6"
        }"#;
        let config: ConsensusConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.models.len(), 2);
        assert_eq!(config.strategy, ConsensusStrategy::BestOf);
        assert_eq!(config.judge, Some("claude-sonnet-4-6".to_string()));
    }

    #[test]
    fn test_consensus_config_deserialize_no_judge() {
        let json = r#"{
            "models": ["gpt-4o"],
            "strategy": "merge"
        }"#;
        let config: ConsensusConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.strategy, ConsensusStrategy::Merge);
        assert!(config.judge.is_none());
    }

    #[test]
    fn test_consensus_metadata_serialize() {
        let meta = ConsensusMetadata {
            strategy: "best_of".to_string(),
            models_queried: vec!["a".to_string(), "b".to_string()],
            judge: Some("a".to_string()),
            winner: Some("b".to_string()),
            individual_latencies_ms: HashMap::from([
                ("a".to_string(), 100),
                ("b".to_string(), 200),
            ]),
        };
        let json = serde_json::to_value(&meta).unwrap();
        assert_eq!(json["strategy"], "best_of");
        assert_eq!(json["winner"], "b");
    }

    #[test]
    fn test_chat_request_with_consensus_deserialize() {
        let json = r#"{
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "consensus": {
                "models": ["gpt-4o", "claude-sonnet-4-6"],
                "strategy": "majority"
            }
        }"#;
        let req: ChatRequest = serde_json::from_str(json).unwrap();
        assert!(req.consensus.is_some());
        let consensus = req.consensus.unwrap();
        assert_eq!(consensus.strategy, ConsensusStrategy::Majority);
        assert_eq!(consensus.models.len(), 2);
    }

    #[test]
    fn test_chat_request_without_consensus_deserialize() {
        let json = r#"{
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}]
        }"#;
        let req: ChatRequest = serde_json::from_str(json).unwrap();
        assert!(req.consensus.is_none());
    }
}
