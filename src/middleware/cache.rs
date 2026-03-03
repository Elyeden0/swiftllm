use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::providers::types::{ChatRequest, ChatResponse};

/// LRU cache entry with TTL
struct CacheEntry {
    response: ChatResponse,
    created_at: Instant,
    last_accessed: Instant,
}

/// A simple LRU cache for non-streaming chat completions
pub struct ResponseCache {
    entries: Mutex<HashMap<u64, CacheEntry>>,
    max_size: usize,
    ttl: Duration,
}

impl ResponseCache {
    pub fn new(max_size: usize, ttl_seconds: u64) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            max_size,
            ttl: Duration::from_secs(ttl_seconds),
        }
    }

    /// Look up a cached response for the given request
    pub fn get(&self, request: &ChatRequest) -> Option<ChatResponse> {
        let key = hash_request(request);
        let mut entries = self.entries.lock().unwrap();

        if let Some(entry) = entries.get_mut(&key) {
            // Check TTL
            if entry.created_at.elapsed() > self.ttl {
                entries.remove(&key);
                return None;
            }
            entry.last_accessed = Instant::now();
            Some(entry.response.clone())
        } else {
            None
        }
    }

    /// Store a response in the cache
    pub fn put(&self, request: &ChatRequest, response: ChatResponse) {
        let key = hash_request(request);
        let mut entries = self.entries.lock().unwrap();

        // Evict oldest entry if at capacity
        if entries.len() >= self.max_size && !entries.contains_key(&key) {
            if let Some(oldest_key) = entries
                .iter()
                .min_by_key(|(_, e)| e.last_accessed)
                .map(|(k, _)| *k)
            {
                entries.remove(&oldest_key);
            }
        }

        entries.insert(
            key,
            CacheEntry {
                response,
                created_at: Instant::now(),
                last_accessed: Instant::now(),
            },
        );
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let entries = self.entries.lock().unwrap();
        CacheStats {
            size: entries.len(),
            max_size: self.max_size,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CacheStats {
    pub size: usize,
    pub max_size: usize,
}

/// Hash a ChatRequest for cache key purposes
fn hash_request(request: &ChatRequest) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    request.model.hash(&mut hasher);
    for msg in &request.messages {
        msg.role.hash(&mut hasher);
        msg.content.hash(&mut hasher);
    }
    // Include sampling params that affect output
    if let Some(t) = request.temperature {
        t.to_bits().hash(&mut hasher);
    }
    if let Some(p) = request.top_p {
        p.to_bits().hash(&mut hasher);
    }
    if let Some(max) = request.max_tokens {
        max.hash(&mut hasher);
    }
    hasher.finish()
}
