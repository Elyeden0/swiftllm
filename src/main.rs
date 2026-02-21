mod config;

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "llm-proxy")]
#[command(about = "A blazing-fast universal LLM gateway")]
struct Cli {
    /// Path to config file
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,

    /// Port to listen on (overrides config)
    #[arg(short, long)]
    port: Option<u16>,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let mut config = config::Config::load(&cli.config)?;

    if let Some(port) = cli.port {
        config.port = port;
    }

    println!(
        "llm-proxy: loaded {} provider(s), port {}",
        config.providers.len(),
        config.port
    );

    // TODO: initialize tracing/logging
    // TODO: build provider instances from config
    // TODO: start HTTP server

    Ok(())
}
