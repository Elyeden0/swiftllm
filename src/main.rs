mod config;
mod providers;

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "llm-proxy")]
#[command(about = "A blazing-fast universal LLM gateway")]
struct Cli {
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,
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

    // TODO: build provider instances
    // TODO: start HTTP server

    Ok(())
}
