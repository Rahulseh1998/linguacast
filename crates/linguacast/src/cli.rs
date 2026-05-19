use clap::Parser;
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "linguacast",
    version,
    about = "Dub a video into other languages in the speaker's own voice.",
    long_about = "linguacast input.mp4 --langs es,zh,hi,fr,de,ja,pt,ar — local-first by default."
)]
pub struct Args {
    /// Source video file (any format ffmpeg can read).
    pub input: PathBuf,

    /// Comma-separated list of target language codes (BCP-47 / ISO 639-1).
    /// Week-1 spike supports `es` only; other codes are wired in but will error.
    #[arg(long, value_delimiter = ',', default_value = "es")]
    pub langs: Vec<Lang>,

    /// Output directory. Defaults to `./linguacast-out/`.
    #[arg(long, default_value = "linguacast-out")]
    pub out_dir: PathBuf,

    /// Force a compute device. Default: auto-detect (mps → cuda → cpu).
    #[arg(long)]
    pub device: Option<Device>,

    /// Path to the Python sidecar interpreter. Default: looks for `python3`
    /// in PATH and the venv at `sidecar/.venv/bin/python` next to the binary.
    #[arg(long)]
    pub python: Option<PathBuf>,

    /// Override the TTS engine. Default selected per platform: qwen3-tts on
    /// machines with ≥12 GB unified memory, qwen3-tts-quantized otherwise.
    /// Voicebox is wired but disabled until license clears the Apache/MIT floor.
    #[arg(long)]
    pub tts: Option<TtsEngine>,

    /// Skip the consent-gate prompt. Placeholder for v0; the real gate
    /// lands in week 3 (OPE-12) and refuses voice-clone output without it.
    #[arg(long, hide = true)]
    pub i_understand_voice_clone_risks: bool,

    /// Verbose logging (RUST_LOG=debug equivalent).
    #[arg(long, short)]
    pub verbose: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lang(pub String);

impl FromStr for Lang {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let normalized = s.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            return Err("language code cannot be empty".into());
        }
        // Loose validation — full BCP-47 is overkill for the spike. We just
        // reject anything that isn't 2..=8 ASCII alpha so junk fails early.
        if !normalized
            .chars()
            .all(|c| c.is_ascii_alphabetic() || c == '-')
            || normalized.len() < 2
            || normalized.len() > 8
        {
            return Err(format!("invalid language code: {s}"));
        }
        Ok(Lang(normalized))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
#[clap(rename_all = "lowercase")]
pub enum Device {
    Auto,
    Mps,
    Cuda,
    Cpu,
}

#[derive(Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum TtsEngine {
    Qwen3Tts,
    Qwen3TtsQuantized,
    Voicebox,
}
