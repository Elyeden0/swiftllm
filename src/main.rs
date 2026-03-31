use swiftllm::config;
use swiftllm::server;

use clap::Parser;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

#[derive(Parser)]
#[command(name = "swiftllm")]
#[command(about = "A blazing-fast universal LLM gateway")]
#[command(version)]
struct Cli {
    /// Path to .env file (default: looks next to the executable, then current directory)
    #[arg(short = 'e', long = "env")]
    env_file: Option<PathBuf>,

    /// Port to listen on (overrides .env)
    #[arg(short, long)]
    port: Option<u16>,
}

/// Find the .env file path, searching next to the executable first, then the current directory.
fn find_env_file(cli_path: Option<&PathBuf>) -> Option<PathBuf> {
    // 1. Explicit CLI argument takes priority
    if let Some(path) = cli_path {
        if path.exists() {
            return Some(path.clone());
        }
        return None;
    }

    // 2. Look next to the executable (the primary expected location)
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let env_path = exe_dir.join(".env");
            if env_path.exists() {
                return Some(env_path);
            }
        }
    }

    // 3. Fall back to current working directory
    let cwd_env = PathBuf::from(".env");
    if cwd_env.exists() {
        return Some(cwd_env);
    }

    None
}

/// Initialize OpenTelemetry tracing pipeline and return a shutdown guard.
/// When the guard is dropped, it flushes and shuts down the tracer provider.
fn init_otel(
    otel_config: &config::OtelConfig,
) -> Option<opentelemetry_sdk::trace::SdkTracerProvider> {
    if !otel_config.enabled {
        return None;
    }

    use opentelemetry_otlp::WithExportConfig;

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&otel_config.endpoint)
        .build()
        .expect("Failed to create OTLP exporter");

    let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(
            opentelemetry_sdk::Resource::builder()
                .with_service_name(otel_config.service_name.clone())
                .build(),
        )
        .build();

    Some(provider)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Find and load the .env file
    let env_path = find_env_file(cli.env_file.as_ref());

    match env_path {
        Some(ref path) => {
            dotenvy::from_path(path).map_err(|e| {
                anyhow::anyhow!("Failed to load .env file at {}: {}", path.display(), e)
            })?;
        }
        None => {
            let exe_dir = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.display().to_string()))
                .unwrap_or_else(|| "unknown".to_string());

            eprintln!();
            eprintln!("  ERROR: No .env file found!");
            eprintln!();
            eprintln!("  swiftllm requires a .env file to run. Searched in:");
            eprintln!("    1. Next to the executable: {}/.env", exe_dir);
            eprintln!(
                "    2. Current directory: {}/.env",
                std::env::current_dir()
                    .map(|d| d.display().to_string())
                    .unwrap_or_else(|_| ".".to_string())
            );
            eprintln!();
            eprintln!("  Create a .env file with your provider API keys.");
            eprintln!("  See .env.example for a template.");
            eprintln!();
            std::process::exit(1);
        }
    }

    // Load config from environment variables
    let mut config = config::Config::load_from_env()?;

    // CLI port overrides config port
    if let Some(port) = cli.port {
        config.port = port;
    }

    // Initialize OpenTelemetry (must happen before tracing subscriber init)
    let otel_provider = init_otel(&config.otel);

    // Build the tracing subscriber with optional OpenTelemetry layer
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "swiftllm=info".into());

    let fmt_layer = tracing_subscriber::fmt::layer();

    let registry = tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer);

    if let Some(ref provider) = otel_provider {
        use opentelemetry::trace::TracerProvider;
        let tracer = provider.tracer("swiftllm");
        let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
        registry.with(otel_layer).init();
    } else {
        registry.init();
    }

    if let Some(ref path) = env_path {
        info!("Loaded .env from {}", path.display());
    }

    let port = config.port;
    let provider_count = config.providers.len();
    let cache_enabled = config.cache.enabled;
    let cache_max = config.cache.max_size;
    let cache_ttl = config.cache.ttl_seconds;
    let otel_enabled = config.otel.enabled;

    // Build app state and router
    let state = Arc::new(server::AppState::new(config));
    let app = server::build_router(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!(
        "swiftllm v{} starting on http://{}",
        env!("CARGO_PKG_VERSION"),
        addr
    );
    info!("{} provider(s) configured", provider_count);
    if cache_enabled {
        info!(
            "Cache enabled (max {} entries, {}s TTL)",
            cache_max, cache_ttl
        );
    }
    if otel_enabled {
        info!("OpenTelemetry tracing enabled");
    }
    info!("Dashboard: http://{}/dashboard", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    // Graceful shutdown: flush any remaining spans
    if let Some(provider) = otel_provider {
        provider.shutdown().ok();
    }

    Ok(())
}
