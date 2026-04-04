![SwiftLLM](https://img.shields.io/badge/SwiftLLM-v0.4.0-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![crates.io](https://img.shields.io/crates/v/swiftllm.svg)](https://crates.io/crates/swiftllm)
[![Docker](https://img.shields.io/badge/Docker-Available-2496ED.svg)](https://www.docker.com/)
[![npm](https://img.shields.io/badge/npm-swiftllm-CB3837.svg)](https://www.npmjs.com/package/swiftllm)
[![PyPI](https://img.shields.io/badge/PyPI-swiftllm-3775A9.svg)](https://pypi.org/project/swiftllm/)

# SwiftLLM

The universal LLM gateway. Write once, route to **100+ providers** in **23 languages**.

SwiftLLM unifies OpenAI, Anthropic, Gemini, Mistral, Ollama, Groq, Together AI, AWS Bedrock, and 100+ additional providers (DeepSeek, xAI/Grok, Perplexity, Cohere, AI21, Cerebras, and more) behind a single, blazing-fast API. Route intelligently with cost and latency optimization, run multi-model consensus, cache responses, and scale with built-in rate limiting—all without changing your code.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                 Your App                                    │
└────────────────────┬────────────────────────────────────────────────────────┘
                     │
                     │ OpenAI-compatible API
                     │
        ┌────────────▼─────────────┐
        │   SwiftLLM Gateway       │
        │  (Smart Routing, Cache,  │
        │   Consensus, Rate Limit) │
        └────────────┬─────────────┘
                     │
    ┌────────────────┼────────────────┬──────────────────┬──────────────┐
    │                │                │                  │              │
    ▼                ▼                ▼                  ▼              ▼
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐   ┌──────────┐
│ OpenAI  │    │Anthropic │    │ Gemini   │    │   Mistral    │   │  Ollama  │
│(GPT-4o) │    │(Claude)  │    │(Gemini2) │    │              │   │ (Local)  │
└─────────┘    └──────────┘    └──────────┘    └──────────────┘   └──────────┘
    │                │                │                  │              │
    └────────────────┴────────────────┴──────────────────┴──────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
            ┌────────────────┐        ┌──────────────────┐
            │ Groq, Together │        │ AWS Bedrock, &   │
            │ AI, & More     │        │ 85+ Compatible   │
            └────────────────┘        └──────────────────┘
```

## Features

- **100+ Providers**: First-class support for OpenAI, Anthropic, Gemini, Mistral, Ollama, Groq, Together AI, AWS Bedrock, plus 100+ OpenAI-compatible providers (DeepSeek, xAI/Grok, Perplexity, Cohere, AI21, Cerebras, SambaNova, Lambda Labs, etc.)
- **Smart Routing**: Route requests to the best model by cost (cost-optimized), latency (latency-optimized), or both (balanced)
- **Quality Tiers**: Specify `low`, `medium`, or `high` quality—SwiftLLM picks the right model automatically
- **Multi-Model Consensus**: Query 3+ models in parallel and combine responses using `best_of`, `majority`, or `merge` strategies
- **Response Caching**: LRU cache with configurable TTL (time-to-live)
- **Rate Limiting**: Per-provider rate limiting with automatic backoff
- **Cost Tracking**: Track tokens and compute estimated costs per provider/model
- **Automatic Failover**: Priority chains let traffic fall back to secondary providers on error
- **SSE Streaming**: Full streaming support with automatic format translation between providers
- **OpenTelemetry**: GenAI semantic conventions on all spans for observability
- **Tower Middleware**: Built on Axum, Tower, and Tokio for production reliability
- **MCP Server**: `swiftllm-mcp` binary with 7 tools for Claude integration
- **Docker Ready**: Single-command deployment with embedded binaries
- **23 Language Bindings**: Rust, Python, Node.js/TypeScript, C, C++, Go, Java, C#, Ruby, PHP, Elixir, Swift, Kotlin, Zig, Scala, Haskell, OCaml, Lua, Perl, D, Nim, Dart, Erlang

## Quick Start

### Install (Cargo)

```bash
cargo install swiftllm
```

### Docker

```bash
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=sk-... \
  -e OPENAI_MODELS=gpt-4o,gpt-4o-mini \
  ghcr.io/elyeden0/swiftllm:latest
```

### Binary Download

Download prebuilt binaries from [GitHub Releases](https://github.com/Elyeden0/swiftllm/releases).

### Quick Test

```bash
# Start the server
PORT=8080 OPENAI_API_KEY=sk-... swiftllm-server

# In another terminal, test it
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello, world!"}]
  }'
```

## Supported Providers

### Tier 1: First-Class Support

| Provider | Models | Type |
|----------|--------|------|
| **OpenAI** | GPT-4o, GPT-4.1, o3, o3-mini, o4-mini | Commercial |
| **Anthropic** | Claude Sonnet, Claude Opus, Claude Haiku | Commercial |
| **Google Gemini** | Gemini 2.0 Flash, Gemini 2.0 Pro, Gemini 1.5 | Commercial |
| **Mistral** | Mistral Large, Small, Codestral, Pixtral, Ministral | Commercial |
| **Ollama** | Llama, Mistral, CodeLlama, and 100+ local models | Local |
| **Groq** | Llama 3.1, Mixtral, Gemma | Commercial |
| **Together AI** | OpenLLaMA, Falcon, Qwen, and 200+ open-source models | Commercial |
| **AWS Bedrock** | Claude, Llama, Cohere, AI21, Mistral | Commercial |

### Tier 2: 100+ OpenAI-Compatible Providers

Automatically supported via `GenericProvider`:

- **Major Cloud**: DeepSeek, xAI (Grok), Perplexity, Cohere, AI21, Cerebras, SambaNova, Lambda Labs
- **Local/Self-Hosted**: Ollama, vLLM, LM Studio, Hugging Face Text Generation Inference
- **Specialized**: Fireworks AI, NVIDIA NIM, Baseten, Anyscale, Azure OpenAI, and 80+ more

Configure any OpenAI-compatible provider with:

```yaml
[providers.myai]
kind = "Generic"
base_url = "https://api.provider.ai/v1"
api_key = "sk-..."
models = ["model-a", "model-b"]
```

## Smart Routing

Let SwiftLLM choose the best model based on your constraints:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Summarize this..."}],
    "routing": {
      "strategy": "cost_optimized",
      "quality": "medium",
      "max_cost_per_1k_tokens": 0.01
    }
  }'
```

### Routing Strategies

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| `cost_optimized` | Pick the cheapest model that meets quality tier | High-volume, latency-tolerant workloads |
| `latency_optimized` | Pick the fastest model that meets quality tier | Real-time chat, low-latency APIs |
| `balanced` | Minimize cost × latency (Pareto frontier) | General-purpose applications |

### Quality Tiers

- **Low**: Fast, cheap models (e.g., GPT-4o-mini, Claude Haiku)
- **Medium**: Balanced performance and cost (e.g., GPT-4o, Claude Sonnet)
- **High**: Best-in-class models (e.g., Claude Opus, o4-mini)

Response includes `routing_metadata` with the chosen model and reasoning.

## Multi-Model Consensus

Query multiple models and combine results:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ignored",
    "messages": [{"role": "user", "content": "Explain quantum computing..."}],
    "consensus": {
      "models": ["gpt-4o", "claude-opus-4-6", "gemini-2.0-pro"],
      "strategy": "merge",
      "judge": "gpt-4o"
    }
  }'
```

### Consensus Strategies

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| `best_of` | Query all models, score with judge, return best | Critical decisions, fact-checking |
| `majority` | Query all models, return most common response | Voting-based approaches |
| `merge` | Query all models, have judge synthesize one response | Combine strengths of multiple models |

Response includes `consensus_metadata` with models queried and latencies.

## Language Bindings

SwiftLLM is accessible from **23 languages** via C FFI and language-specific wrappers:

| Language | Installation | Example |
|----------|--------------|---------|
| **Rust** | `cargo add swiftllm` | `swiftllm::init()` |
| **Python** | `pip install swiftllm` | `import swiftllm` |
| **Node.js/TypeScript** | `npm install swiftllm` | `import { swiftllm } from 'swiftllm'` |
| **C** | Link `libswiftllm.so` | `swiftllm_init()` |
| **C++** | Link `libswiftllm.so` | `SwiftLLM::init()` |
| **Go** | `go get github.com/elyeden0/swiftllm-go` | `swiftllm.Init()` |
| **Java** | `maven: com.github.elyeden0:swiftllm` | `SwiftLLM.init()` |
| **C#** | `dotnet add package SwiftLLM` | `SwiftLLM.Init()` |
| **Ruby** | `gem install swiftllm` | `require 'swiftllm'` |
| **PHP** | `composer require elyeden0/swiftllm` | `Swiftllm\init()` |
| **Elixir** | `mix add swiftllm` | `:swiftllm.init()` |
| **Swift** | `swift package add swiftllm` | `Swiftllm.init()` |
| **Kotlin** | `gradle: com.github.elyeden0:swiftllm` | `Swiftllm.init()` |
| **Zig** | `build.zig` | `swiftllm.init()` |
| **Scala** | `sbt: "com.github.elyeden0" %% "swiftllm"` | `SwiftLLM.init()` |
| **Haskell** | `cabal: swiftllm` | `SwiftLLM.init` |
| **OCaml** | `opam: swiftllm` | `Swiftllm.init ()` |
| **Lua** | `luarocks install swiftllm` | `swiftllm.init()` |
| **Perl** | `cpan: Swiftllm` | `Swiftllm::init()` |
| **D** | `dub: swiftllm` | `swiftllm.init()` |
| **Nim** | `nimble install swiftllm` | `swiftllm.init()` |
| **Dart** | `pub add swiftllm` | `swiftllm.init()` |
| **Erlang** | `rebar3: {swiftllm, "..."}` | `swiftllm:init()` |

All bindings use the same C FFI surface, ensuring consistent behavior across languages.

## MCP Server

SwiftLLM includes a Model Context Protocol (MCP) server that exposes 7 tools for Claude and other MCP clients:

### Run the MCP Server

```bash
swiftllm-mcp --config /path/to/config.toml
```

Then configure your MCP client (e.g., in Claude):

```json
{
  "mcpServers": {
    "swiftllm": {
      "command": "swiftllm-mcp",
      "args": ["--config", "/path/to/config.toml"]
    }
  }
}
```

### Available Tools

1. **chat_completion** – Send a chat message to any configured LLM
2. **list_models** – Show all available models across providers
3. **list_providers** – Show configured LLM providers
4. **smart_route** – Use cost/latency routing to pick and query a model
5. **consensus_query** – Query multiple models and combine with a strategy
6. **get_stats** – View usage statistics (requests, tokens, cost, cache hits)
7. **compare_models** – Run the same prompt against multiple models side-by-side

## Docker

### Single Container

```bash
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=sk-... \
  -e OPENAI_MODELS=gpt-4o,gpt-4o-mini \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e ANTHROPIC_MODELS=claude-opus-4-6,claude-haiku-4-5-20251001 \
  ghcr.io/elyeden0/swiftllm:latest
```

### Docker Compose

```yaml
version: '3.8'

services:
  swiftllm:
    image: ghcr.io/elyeden0/swiftllm:latest
    ports:
      - "8080:8080"
    environment:
      PORT: 8080
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      OPENAI_MODELS: gpt-4o,gpt-4o-mini
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      ANTHROPIC_MODELS: claude-opus-4-6,claude-haiku-4-5-20251001
      CACHE_ENABLED: "true"
      CACHE_TTL_SECONDS: "300"
      RATE_LIMIT_ENABLED: "true"
    volumes:
      - ./.env:/home/swiftllm/.env:ro
```

Start with:

```bash
docker-compose up -d
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion with optional routing/consensus |
| `/v1/embeddings` | POST | Generate embeddings (provider-dependent) |
| `/v1/models` | GET | List all available models |
| `/health` | GET | Health check (returns `{"status": "ok"}`) |
| `/api/stats` | GET | View usage stats, cost, cache, rate limits |
| `/dashboard` | GET | Embedded web dashboard (HTML) |

### Example: Chat Completion

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "system", "content": "You are helpful."},
      {"role": "user", "content": "What is 2+2?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100,
    "stream": false
  }'
```

### Example: Streaming

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Tell me a story..."}],
    "stream": true
  }'
```

## Configuration

Create a `.env` file in the same directory as `swiftllm-server`, or set environment variables directly.

### Server

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | Server port |
| `AUTH_API_KEYS` | (empty) | Comma-separated bearer tokens (leave empty to disable auth) |

### Cache

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_ENABLED` | `true` | Enable response caching |
| `CACHE_MAX_SIZE` | `1000` | Max cached responses |
| `CACHE_TTL_SECONDS` | `300` | Cache entry time-to-live (5 minutes) |

### Rate Limiting

| Variable | Default | Description |
|----------|---------|-------------|
| `RATE_LIMIT_ENABLED` | `true` | Enable rate limiting |
| `RATE_LIMIT_MAX_REQUESTS` | `100` | Default max requests per window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Window duration (per-provider overrides available) |

### Providers

For each provider, set:

```
{PROVIDER}_API_KEY=sk-...
{PROVIDER}_MODELS=model-a,model-b,model-c
{PROVIDER}_PRIORITY=1
{PROVIDER}_BASE_URL=https://api.custom.ai/v1  # optional
```

Example:

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODELS=gpt-4o,gpt-4o-mini
OPENAI_PRIORITY=1

ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODELS=claude-opus-4-6
ANTHROPIC_PRIORITY=2

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODELS=llama3:latest,mistral:latest
OLLAMA_PRIORITY=10
```

### Observability

| Variable | Default | Description |
|----------|---------|-------------|
| `RUST_LOG` | `info` | Tracing level: `debug`, `info`, `warn`, `error` |
| `OTEL_ENABLED` | `false` | Enable OpenTelemetry |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | (empty) | OTLP collector endpoint (e.g., `http://localhost:4317`) |

Full `.env` example: see `.env.example` in the repository.

## Comparison with Other Gateways

| Feature | SwiftLLM | LiteLLM | Liter-LLM |
|---------|----------|---------|-----------|
| **Providers** | 100+ (8 first-class + 92 compatible) | 100+ | 50+ |
| **Language Bindings** | 23 (Rust, Python, Node, Go, Java, C#, Ruby, PHP, Swift, Kotlin, Zig, Scala, Haskell, OCaml, Lua, Perl, D, Nim, Dart, Erlang, C, C++, Elixir) | 2 (Python, TypeScript) | Python only |
| **Smart Routing** | 3 strategies + quality tiers | Basic proxy pass-through | Manual routing |
| **Multi-Model Consensus** | 3 strategies with judge model | No | No |
| **Streaming** | Full SSE with translation | Yes | Yes |
| **Caching** | LRU with TTL | Via LiteLLM proxy | No |
| **Rate Limiting** | Per-provider with backoff | Basic | Basic |
| **MCP Server** | 7 tools, Claude-ready | No | No |
| **OpenTelemetry** | GenAI conventions | Limited | No |
| **Speed** | Rust (native) | Python (slower) | Python (slower) |
| **Docker** | Single binary | Python + dependencies | Python + dependencies |
| **License** | MIT | MIT | MIT |

## License

MIT License. See [LICENSE](LICENSE) file for details.

---

## Getting Help

- **GitHub Issues**: [github.com/Elyeden0/swiftllm/issues](https://github.com/Elyeden0/swiftllm/issues)
- **GitHub Discussions**: [github.com/Elyeden0/swiftllm/discussions](https://github.com/Elyeden0/swiftllm/discussions)
- **Documentation**: [Full API docs](https://github.com/Elyeden0/swiftllm#api-reference)

---

**Made with ❤️ in Rust**
