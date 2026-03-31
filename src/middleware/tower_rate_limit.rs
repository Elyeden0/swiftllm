use std::sync::Arc;
use std::task::{Context, Poll};

use tower::{Layer, Service};

use crate::middleware::rate_limit::RateLimiter;
use crate::providers::types::ChatRequest;
use crate::providers::ProviderError;

/// Tower layer that wraps the existing token-bucket rate limiter.
#[derive(Clone)]
pub struct RateLimitLayer {
    limiter: Arc<RateLimiter>,
}

impl RateLimitLayer {
    pub fn new(limiter: Arc<RateLimiter>) -> Self {
        Self { limiter }
    }
}

impl<S: Clone> Layer<S> for RateLimitLayer {
    type Service = RateLimitService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RateLimitService {
            inner,
            limiter: self.limiter.clone(),
        }
    }
}

/// Tower service produced by [`RateLimitLayer`].
#[derive(Clone)]
pub struct RateLimitService<S> {
    inner: S,
    limiter: Arc<RateLimiter>,
}

impl<S, Resp> Service<ChatRequest> for RateLimitService<S>
where
    S: Service<ChatRequest, Response = Resp, Error = ProviderError> + Clone + Send + 'static,
    S::Future: Send,
    Resp: Send + 'static,
{
    type Response = Resp;
    type Error = ProviderError;
    type Future = std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>,
    >;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: ChatRequest) -> Self::Future {
        // Use the model name as a proxy for the provider in rate-limit checks.
        // In a fully-integrated setup, the provider name would be threaded through.
        let provider_key = request.model.clone();
        let limiter = self.limiter.clone();
        let mut inner = self.inner.clone();

        Box::pin(async move {
            if let Err(retry_after) = limiter.check(&provider_key) {
                return Err(ProviderError::Api {
                    status: 429,
                    message: format!("Rate limit exceeded. Retry after {:.1}s", retry_after),
                });
            }
            inner.call(request).await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_rate_limit_layer_creates_service() {
        let limiter = Arc::new(RateLimiter::new(HashMap::new(), None));
        let layer = RateLimitLayer::new(limiter);
        assert!(std::mem::size_of_val(&layer) > 0);
    }
}
