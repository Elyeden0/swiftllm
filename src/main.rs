mod config;
mod middleware;
mod providers;
mod server;

use clap::Parser;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

#[derive(Parser)]
#[command(name = "llm-proxy")]
#[command(about = "A blazing-fast universal LLM gateway")]
#[command(version)]
struct Cli {
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,
    #[arg(short, long)]
    port: Option<u16>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "llm_proxy=info".into()),
        )
        .init();

    let cli = Cli::parse();
    let mut config = config::Config::load(&cli.config)?;

    if let Some(port) = cli.port {
        config.port = port;
    }

    let port = config.port;
    let provider_count = config.providers.len();
    let cache_enabled = config.cache.enabled;
    let cache_max = config.cache.max_size;
    let cache_ttl = config.cache.ttl_seconds;

    let state = Arc::new(server::AppState::new(config));
    let app = server::build_router(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("llm-proxy v{} starting on http://{}", env!("CARGO_PKG_VERSION"), addr);
    info!("{} provider(s) configured", provider_count);
    if cache_enabled {
        info!("Cache enabled (max {} entries, {}s TTL)", cache_max, cache_ttl);
    }
    // TODO: add dashboard URL log

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
