use std::sync::Arc;
use std::task::{Context, Poll};

use tower::{Layer, Service};

use crate::middleware::cache::ResponseCache;
use crate::providers::types::{ChatRequest, ChatResponse};
use crate::providers::ProviderError;

/// Tower layer that wraps the existing LRU cache logic.
///
/// When applied, non-streaming requests are checked against the cache before
/// being forwarded to the inner service. Successful responses are stored.
#[derive(Clone)]
pub struct CacheLayer {
    cache: Arc<ResponseCache>,
}

impl CacheLayer {
    pub fn new(cache: Arc<ResponseCache>) -> Self {
        Self { cache }
    }
}

impl<S: Clone> Layer<S> for CacheLayer {
    type Service = CacheService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        CacheService {
            inner,
            cache: self.cache.clone(),
        }
    }
}

/// Tower service produced by [`CacheLayer`].
#[derive(Clone)]
pub struct CacheService<S> {
    inner: S,
    cache: Arc<ResponseCache>,
}

impl<S> Service<ChatRequest> for CacheService<S>
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
        // Only cache non-streaming requests
        if !request.is_streaming() {
            if let Some(cached) = self.cache.get(&request) {
                return Box::pin(async move { Ok(cached) });
            }
        }

        let cache = self.cache.clone();
        let mut inner = self.inner.clone();
        let is_streaming = request.is_streaming();

        Box::pin(async move {
            let response = inner.call(request.clone()).await?;
            if !is_streaming {
                cache.put(&request, response.clone());
            }
            Ok(response)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_layer_creates_service() {
        let cache = Arc::new(ResponseCache::new(10, 60));
        let layer = CacheLayer::new(cache);
        // Verify the layer can be constructed without panicking
        assert!(std::mem::size_of_val(&layer) > 0);
    }
}
