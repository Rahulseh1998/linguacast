use anyhow::Result;
use clap::Parser;

mod cli;
mod device;
mod ffmpeg;
mod pipeline;
mod sidecar;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .compact()
        .init();

    let cli = cli::Cli::parse();
    match cli.command.clone() {
        Some(cli::Command::Pull {
            asr,
            mt,
            tts_size,
            python,
        }) => pipeline::run_pull(pipeline::PullOpts {
            asr,
            mt,
            tts_size,
            python,
        }),
        None => pipeline::run_dub(cli),
    }
}
