use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "linguacast",
    version,
    about = "Dub a video into other languages in the speaker's own voice.",
    long_about = "linguacast input.mp4 --langs es,zh,hi,fr,de,ja,pt,ar — local-first by default.\n\
                  Run `linguacast pull` first to pre-download model weights (~10 GB)."
)]
pub struct Cli {
    /// Subcommand. If omitted, the binary defaults to dub mode and the
    /// remaining args (input file, --langs, etc.) apply.
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Source video file (any format ffmpeg can read). Required in default
    /// (dub) mode; ignored when a subcommand is given.
    pub input: Option<PathBuf>,

    /// Comma-separated list of target language codes (BCP-47 / ISO 639-1).
    /// Week-1 spike supports `es` only; other codes are wired in but will
    /// error this week.
    #[arg(long, value_delimiter = ',', default_value = "es")]
    pub langs: Vec<Lang>,

    /// Output directory. Defaults to `./linguacast-out/`.
    #[arg(long, default_value = "linguacast-out")]
    pub out_dir: PathBuf,

    /// Force a compute device. Default: auto-detect (mps → cuda → cpu).
    #[arg(long)]
    pub device: Option<Device>,

    /// Path to the Python sidecar interpreter. Default: the venv at
    /// `sidecar/.venv/bin/python` next to the binary, falling back to
    /// `python3` from PATH.
    #[arg(long)]
    pub python: Option<PathBuf>,

    /// Override the ASR (speech-to-text) model. Default: `large-v3`.
    /// Pre-approved fallback per the OPE-19 CTO ack: `medium` if
    /// `large-v3` doesn't fit the 8 GB ceiling.
    #[arg(long, default_value = "large-v3")]
    pub asr: AsrModel,

    /// Override the MT (translation) model. Default: `m2m100-418m`
    /// (facebook/m2m100_418M, MIT, ~5 GB peak — fits 8 GB M1). Opt-in
    /// `madlad-3b` is available for ≥16 GB hosts. See
    /// `docs/engine-decision.md` for the fit measurements.
    #[arg(long, default_value = "m2m100-418m")]
    pub mt: MtModel,

    /// Override the TTS engine. Default: qwen3-tts. Voicebox is wired but
    /// disabled until license clears the Apache/MIT floor.
    #[arg(long)]
    pub tts: Option<TtsEngine>,

    /// Qwen3-TTS variant size. Default: 1.7B. Use `0.6B` on tighter boxes
    /// (<12 GB unified memory).
    #[arg(long, default_value = "1.7B")]
    pub tts_size: TtsSize,

    /// Skip the consent-gate prompt. Placeholder for v0; the real gate
    /// lands in week 3 (OPE-12) and refuses voice-clone output without it.
    #[arg(long, hide = true)]
    pub i_understand_voice_clone_risks: bool,

    /// Verbose logging (RUST_LOG=debug equivalent).
    #[arg(long, short)]
    pub verbose: bool,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Command {
    /// Pre-download all model weights (Whisper, MADLAD, Qwen3-TTS) into
    /// the local cache. Run this once after install so the first dub is
    /// the warm-cache run that the launch-hook TTW measures.
    Pull {
        /// ASR model to pull. Default: `large-v3`.
        #[arg(long, default_value = "large-v3")]
        asr: AsrModel,

        /// MT model to pull. Default: `m2m100-418m`.
        #[arg(long, default_value = "m2m100-418m")]
        mt: MtModel,

        /// TTS size to pull. Default: `1.7B`.
        #[arg(long, default_value = "1.7B")]
        tts_size: TtsSize,

        /// Path to the Python sidecar interpreter (see top-level `--python`).
        #[arg(long)]
        python: Option<PathBuf>,
    },
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

/// Whisper model variant. `large-v3` is the launch default; `medium` is
/// the pre-approved fallback per the OPE-19 CTO ack (smaller resident
/// footprint when large-v3 doesn't fit the 8 GB ceiling).
#[derive(Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum AsrModel {
    LargeV3,
    Medium,
    Small,
}

impl AsrModel {
    pub fn as_str(&self) -> &'static str {
        match self {
            AsrModel::LargeV3 => "large-v3",
            AsrModel::Medium => "medium",
            AsrModel::Small => "small",
        }
    }
}

/// MT model registry. `m2m100-418m` (MIT, ~5 GB peak) is the default and
/// fits the 8 GB M1 ceiling. `madlad-3b` (Apache-2.0, ~6 GB CPU bf16) is
/// the opt-in upgrade for ≥16 GB hosts. NLLB is rejected (CC-BY-NC). The
/// `MADLAD-400-1B` variant referenced in the CTO ack fallback list does
/// not exist as a public Apache-2.0 release (verified 2026-05-19).
#[derive(Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum MtModel {
    M2M100418M,
    Madlad3B,
    Madlad10B,
}

impl MtModel {
    pub fn as_str(&self) -> &'static str {
        match self {
            MtModel::M2M100418M => "m2m100-418m",
            MtModel::Madlad3B => "madlad-3b",
            MtModel::Madlad10B => "madlad-10b",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum TtsSize {
    #[clap(name = "0.6B")]
    Small,
    #[clap(name = "1.7B")]
    Large,
}

impl TtsSize {
    pub fn as_str(&self) -> &'static str {
        match self {
            TtsSize::Small => "0.6B",
            TtsSize::Large => "1.7B",
        }
    }
}
