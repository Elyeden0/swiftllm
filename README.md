# swiftllm

A blazing-fast universal LLM gateway written in Rust. Route requests to OpenAI, Anthropic, Google Gemini, Mistral, Ollama, and more through a single OpenAI-compatible API.

```
┌──────────────┐       ┌───────────┐       ┌──────────┐
│  Your App    │──────▶│ swiftllm  │──────▶│  OpenAI  │
│  (any SDK)   │       │  :8080    │──────▶│ Anthropic│
└──────────────┘       └───────────┘──────▶│  Gemini  │
                                    ──────▶│  Mistral │
                                    ──────▶│  Ollama  │
                                           └──────────┘
```

## Why?

Most teams use multiple LLM providers. That means juggling different SDKs, API formats, and billing dashboards. **swiftllm** gives you:

- **One endpoint** — drop-in replacement for the OpenAI API. Use any SDK or tool that speaks OpenAI format.
- **Automatic routing** — requests route to the right provider based on model name (`gpt-4.1` → OpenAI, `claude-sonnet-4-6` → Anthropic, `gemini-2.0-flash` → Google, `mistral-large-latest` → Mistral, `llama3:latest` → Ollama).
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

# Windows
# Download swiftllm-windows-amd64.zip from the releases page and extract it
```

Then configure and run:

```bash
# Copy the example .env and add your API keys
cp .env.example .env
# Edit .env with your API keys...

./swiftllm
```

The `.env` file must be placed in the same directory as the executable. swiftllm will refuse to start without it.

### From source

```bash
git clone https://github.com/Elyeden0/swiftllm
cd swiftllm
cargo build --release

cp .env.example .env
# Edit .env with your API keys...

./target/release/swiftllm
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

All configuration is done through a `.env` file. See [`.env.example`](.env.example) for all options.

```bash
PORT=8080
AUTH_API_KEYS=your-proxy-api-key
DEFAULT_PROVIDER=openai

OPENAI_API_KEY=sk-...
OPENAI_MODELS=gpt-4o,gpt-4.1,o3,o4-mini
OPENAI_PRIORITY=1

ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODELS=claude-sonnet-4-6,claude-opus-4-6
ANTHROPIC_PRIORITY=2

GEMINI_API_KEY=your-gemini-key
GEMINI_MODELS=gemini-2.0-flash,gemini-2.0-pro
GEMINI_PRIORITY=3

MISTRAL_API_KEY=your-mistral-key
MISTRAL_MODELS=mistral-large-latest,codestral-latest
MISTRAL_PRIORITY=4

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODELS=llama3:latest,mistral:latest
OLLAMA_PRIORITY=10
```

You can also pass the path explicitly: `swiftllm --env /path/to/.env`

### Model routing

Models are routed to providers in this order:

1. **Exact match** — if a model name appears in a provider's `MODELS` list
2. **Prefix match** — `gpt-*` → OpenAI, `claude-*` → Anthropic, `gemini-*` → Google, `mistral-*` → Mistral, `model:tag` → Ollama
3. **Default provider** — the `DEFAULT_PROVIDER` fallback

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
