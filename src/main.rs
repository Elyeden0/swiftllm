mod config;
mod failover;
mod middleware;
mod providers;
mod server;

use clap::Parser;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

#[derive(Parser)]
#[command(name = "swiftllm")]
#[command(about = "A blazing-fast universal LLM gateway")]
#[command(version)]
struct Cli {
    /// Path to config file
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,

    /// Port to listen on (overrides config)
    #[arg(short, long)]
    port: Option<u16>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "swiftllm=info".into()),
        )
        .init();

    let cli = Cli::parse();

    // Load config
    let mut config = config::Config::load(&cli.config)?;

    // CLI port overrides config port
    if let Some(port) = cli.port {
        config.port = port;
    }

    let port = config.port;
    let provider_count = config.providers.len();
    let cache_enabled = config.cache.enabled;
    let cache_max = config.cache.max_size;
    let cache_ttl = config.cache.ttl_seconds;

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
    info!("Dashboard: http://{}/dashboard", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
