use std::sync::Arc;
use std::task::{Context, Poll};

use tower::{Layer, Service};

use crate::middleware::cost::CostTracker;
use crate::providers::types::{ChatRequest, ChatResponse};
use crate::providers::ProviderError;

/// Tower layer that wraps the existing cost tracker.
///
/// After a successful chat completion, the layer records token usage and cost.
#[derive(Clone)]
pub struct CostTrackingLayer {
    tracker: Arc<CostTracker>,
}

impl CostTrackingLayer {
    pub fn new(tracker: Arc<CostTracker>) -> Self {
        Self { tracker }
    }
}

impl<S: Clone> Layer<S> for CostTrackingLayer {
    type Service = CostTrackingService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        CostTrackingService {
            inner,
            tracker: self.tracker.clone(),
        }
    }
}

/// Tower service produced by [`CostTrackingLayer`].
#[derive(Clone)]
pub struct CostTrackingService<S> {
    inner: S,
    tracker: Arc<CostTracker>,
}

impl<S> Service<ChatRequest> for CostTrackingService<S>
where
    S: Service<ChatRequest, Response = ChatResponse, Error = ProviderError>
        + Clone
        + Send
        + 'static,
    S::Future: Send,
{
    type Response = ChatResponse;
    type Error = ProviderError;
    type Future = std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>,
    >;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: ChatRequest) -> Self::Future {
        let tracker = self.tracker.clone();
        let model = request.model.clone();
        let mut inner = self.inner.clone();

        Box::pin(async move {
            match inner.call(request).await {
                Ok(response) => {
                    if let Some(usage) = &response.usage {
                        tracker.record_request(
                            &response.model,
                            &model,
                            usage.prompt_tokens,
                            usage.completion_tokens,
                        );
                    }
                    Ok(response)
                }
                Err(e) => {
                    tracker.record_error(&model);
                    Err(e)
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_tracking_layer_creates_service() {
        let tracker = Arc::new(CostTracker::new());
        let layer = CostTrackingLayer::new(tracker);
        assert!(std::mem::size_of_val(&layer) > 0);
    }
}
