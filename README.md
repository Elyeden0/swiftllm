# swiftllm

A blazing-fast universal LLM gateway written in Rust. Route requests to OpenAI, Anthropic, Google Gemini, Ollama, and more through a single OpenAI-compatible API.

```
┌──────────────┐       ┌───────────┐       ┌──────────┐
│  Your App    │──────▶│ swiftllm  │──────▶│  OpenAI  │
│  (any SDK)   │       │  :8080    │──────▶│ Anthropic│
└──────────────┘       └───────────┘──────▶│  Gemini  │
                                    ──────▶│  Ollama  │
                                           └──────────┘
```

## Why?

Most teams use multiple LLM providers. That means juggling different SDKs, API formats, and billing dashboards. **swiftllm** gives you:

- **One endpoint** — drop-in replacement for the OpenAI API. Use any SDK or tool that speaks OpenAI format.
- **Automatic routing** — requests route to the right provider based on model name (`gpt-4o` → OpenAI, `claude-sonnet-4-6` → Anthropic, `gemini-2.0-flash` → Google, `llama3:latest` → Ollama).
- **Streaming support** — full SSE streaming with format translation across all providers.
- **Single binary** — no runtime dependencies, no Docker required. Just download and run.
- **~1ms overhead** — built in Rust with async I/O. Adds negligible latency.

## Quick Start

### Download pre-built binary

Grab the latest release for your platform from the [Releases page](https://github.com/Elyeden0/swiftllm/releases):

```bash
# Linux
curl -L https://github.com/Elyeden0/swiftllm/releases/latest/download/swiftllm-linux-amd64.tar.gz | tar xz
chmod +x swiftllm

# macOS (Apple Silicon)
curl -L https://github.com/Elyeden0/swiftllm/releases/latest/download/swiftllm-macos-arm64.tar.gz | tar xz
chmod +x swiftllm
```

Then configure and run:

```bash
# Download the example config
curl -O https://raw.githubusercontent.com/Elyeden0/swiftllm/main/config.example.toml
cp config.example.toml config.toml
# Add your API keys...

./swiftllm --config config.toml
```

### From source

```bash
git clone https://github.com/Elyeden0/swiftllm
cd swiftllm
cargo build --release

cp config.example.toml config.toml
# Add your API keys...

./target/release/swiftllm --config config.toml
```

### Usage

Once running, point any OpenAI-compatible client at `http://localhost:8080`:

```bash
# Non-streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "claude-sonnet-4-6",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "stream": true
  }'
```

Works with the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="your-proxy-api-key",
)

# Use any model from any provider
response = client.chat.completions.create(
    model="claude-sonnet-4-6",  # Routes to Anthropic
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## Configuration

See [`config.example.toml`](config.example.toml) for all options.

```toml
port = 8080

[auth]
api_keys = ["your-proxy-api-key"]

[providers.openai]
kind = "openai"
api_key = "sk-..."
models = ["gpt-4o", "gpt-4o-mini"]

[providers.anthropic]
kind = "anthropic"
api_key = "sk-ant-..."
models = ["claude-sonnet-4-6", "claude-opus-4-6"]

[providers.gemini]
kind = "gemini"
api_key = "your-gemini-key"
models = ["gemini-2.0-flash", "gemini-2.0-pro"]

[providers.ollama]
kind = "ollama"
base_url = "http://localhost:11434"
models = ["llama3:latest", "mistral:latest"]
```

### Model routing

Models are routed to providers in this order:

1. **Exact match** — if a model name appears in a provider's `models` list
2. **Prefix match** — `gpt-*` → OpenAI, `claude-*` → Anthropic, `gemini-*` → Google, `model:tag` → Ollama
3. **Default provider** — the `routing.default_provider` fallback

## API Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | Chat completions (streaming & non-streaming) |
| `GET /v1/models` | List all configured models |
| `GET /health` | Health check |
| `GET /api/stats` | Usage stats, cost tracking, cache metrics |
| `GET /dashboard` | Live web dashboard |

## Roadmap

- [x] Response caching (LRU with configurable TTL)
- [x] Cost tracking & token counting dashboard
- [x] Automatic failover with priority chains
- [x] Embedded web dashboard
- [x] Rate limiting per provider
- [x] Google Gemini provider
- [ ] Tool/function call translation
- [ ] Request logging & analytics

## License

MIT
