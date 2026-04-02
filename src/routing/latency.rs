use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

const BUFFER_SIZE: usize = 50;

/// Circular buffer storing the last N request durations for a provider+model pair.
#[derive(Debug, Clone)]
struct LatencyBuffer {
    samples: Vec<Duration>,
    /// Next write position (wraps around).
    cursor: usize,
    /// Total samples ever written (used to know if the buffer has wrapped).
    total: usize,
}

impl LatencyBuffer {
    fn new() -> Self {
        Self {
            samples: Vec::with_capacity(BUFFER_SIZE),
            cursor: 0,
            total: 0,
        }
    }

    fn push(&mut self, d: Duration) {
        if self.samples.len() < BUFFER_SIZE {
            self.samples.push(d);
        } else {
            self.samples[self.cursor] = d;
        }
        self.cursor = (self.cursor + 1) % BUFFER_SIZE;
        self.total += 1;
    }

    fn sorted_samples(&self) -> Vec<Duration> {
        let mut v = self.samples.clone();
        v.sort();
        v
    }

    fn percentile(&self, p: f64) -> Option<Duration> {
        let sorted = self.sorted_samples();
        if sorted.is_empty() {
            return None;
        }
        let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
        Some(sorted[idx.min(sorted.len() - 1)])
    }

    fn p50(&self) -> Option<Duration> {
        self.percentile(50.0)
    }

    fn p95(&self) -> Option<Duration> {
        self.percentile(95.0)
    }
}

/// Thread-safe per-provider latency tracker.
#[derive(Debug, Clone)]
pub struct LatencyTracker {
    inner: Arc<RwLock<HashMap<String, LatencyBuffer>>>,
}

impl Default for LatencyTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl LatencyTracker {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Record a request duration for the given provider.
    pub fn record(&self, provider: &str, duration: Duration) {
        let mut map = self.inner.write().unwrap();
        map.entry(provider.to_string())
            .or_insert_with(LatencyBuffer::new)
            .push(duration);
    }

    /// Get the p50 latency for a provider, or None if no data.
    pub fn p50(&self, provider: &str) -> Option<Duration> {
        let map = self.inner.read().unwrap();
        map.get(provider).and_then(|b| b.p50())
    }

    /// Get the p95 latency for a provider, or None if no data.
    pub fn p95(&self, provider: &str) -> Option<Duration> {
        let map = self.inner.read().unwrap();
        map.get(provider).and_then(|b| b.p95())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_tracker_returns_none() {
        let tracker = LatencyTracker::new();
        assert!(tracker.p50("openai").is_none());
        assert!(tracker.p95("openai").is_none());
    }

    #[test]
    fn single_sample() {
        let tracker = LatencyTracker::new();
        tracker.record("openai", Duration::from_millis(100));
        assert_eq!(tracker.p50("openai"), Some(Duration::from_millis(100)));
        assert_eq!(tracker.p95("openai"), Some(Duration::from_millis(100)));
    }

    #[test]
    fn multiple_samples_ordering() {
        let tracker = LatencyTracker::new();
        for ms in [200, 100, 300, 150, 250] {
            tracker.record("test", Duration::from_millis(ms));
        }
        // sorted: 100, 150, 200, 250, 300
        // p50 index = round(0.5 * 4) = 2 → 200ms
        assert_eq!(tracker.p50("test"), Some(Duration::from_millis(200)));
    }

    #[test]
    fn circular_buffer_wraps() {
        let tracker = LatencyTracker::new();
        // Fill beyond capacity
        for i in 0..60 {
            tracker.record("wrap", Duration::from_millis(i * 10));
        }
        // Should still return valid values (last 50 samples: 100..590 ms)
        let p50 = tracker.p50("wrap").unwrap();
        assert!(p50.as_millis() > 0);
    }
}
