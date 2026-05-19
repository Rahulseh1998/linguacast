use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "linguacast",
    version,
    about = "Dub a video into other languages in the speaker's own voice.",
    long_about = "linguacast input.mp4 --langs es,zh,hi,fr,de,ja,pt,ar,ko,ru,it,tr — local-first by default.\n\
                  Model weights auto-download on first run; run `linguacast pull` to pre-fetch."
)]
pub struct Cli {
    /// Subcommand. If omitted, the binary defaults to dub mode and the
    /// remaining args (input file, --langs, etc.) apply.
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Source video file (any format ffmpeg can read). Required in default
    /// (dub) mode; ignored when a subcommand is given.
    pub input: Option<PathBuf>,

    /// Comma-separated list of target language codes (ISO 639-1). Launch set:
    /// es, zh, hi, fr, de, ja, pt, ar, ko, ru, it, tr. M2M-100 accepts the
    /// full ISO 639-1 set — codes outside the launch list will still
    /// translate but TTS prosody for them is best-effort (Qwen3-TTS Auto path).
    #[arg(long, value_delimiter = ',', default_value = "es")]
    pub langs: Vec<Lang>,

    /// Output directory. Defaults to `./linguacast-out/`.
    #[arg(long, default_value = "linguacast-out")]
    pub out_dir: PathBuf,

    /// After all languages render, package the outputs into
    /// `<out_dir>/<stem>.pack.zip` (one MP4 + thumbnail per language plus
    /// a contact-sheet GIF cycling the outputs). Shareable as a single file.
    #[arg(long)]
    pub pack: bool,

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

    /// Path to a signed consent file (required for non-TTY runs). The file
    /// must contain the verbatim attestation line; see `docs/consent-gate.md`.
    /// In interactive mode the gate prompts; this flag overrides the prompt
    /// for CI / batch pipelines.
    #[arg(long, value_name = "PATH")]
    pub i_have_speaker_consent: Option<PathBuf>,

    /// Optional self-declared speaker name. Used for the refusal-list check
    /// and recorded in the consent ledger. Metadata only — it does not
    /// influence voice-clone fidelity.
    #[arg(long, value_name = "NAME")]
    pub speaker_name: Option<String>,

    /// Override the refusal list (JSON, same schema as the embedded one in
    /// `data/refusal-list.json`). For tests and for ops who want a stricter
    /// list than the upstream default.
    #[arg(long, value_name = "PATH")]
    pub refusal_list: Option<PathBuf>,

    /// Override the consent-record store directory. Default is
    /// `$LINGUACAST_HOME/consents/` or `~/.linguacast/consents/`.
    #[arg(long, value_name = "PATH")]
    pub consent_store_dir: Option<PathBuf>,

    /// Deprecated week-1 flag. The consent gate (OPE-12) is the real check
    /// now. Kept hidden so old scripts still parse without effect.
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
    /// Inspect a (possibly third-party-edited) MP4/M4A and report:
    ///   • whether the LinguaCast perceptual watermark is detected,
    ///   • the watermark id recovered from the audio bits,
    ///   • the consent-hash / version / signer pulled from container metadata.
    /// Per OPE-13, the watermark is the load-bearing claim — metadata can be
    /// stripped by `ffmpeg -map_metadata -1` but the watermark survives a
    /// 1080p H.264 + AAC re-encode at ≥80% on the survival corpus.
    Verify {
        /// Path to the file to inspect (MP4, M4A, WAV — anything ffmpeg can read).
        input: PathBuf,

        /// Path to the Python sidecar interpreter (see top-level `--python`).
        #[arg(long)]
        python: Option<PathBuf>,

        /// Emit a machine-readable JSON report instead of the human one.
        #[arg(long)]
        json: bool,
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
pub enum MtModel {
    #[clap(name = "m2m100-418m")]
    M2M100418M,
    #[clap(name = "madlad-3b")]
    Madlad3B,
    #[clap(name = "madlad-10b")]
    Madlad10B,
    #[clap(name = "helsinki-en-es")]
    HelsinkiEnEs,
}

impl MtModel {
    pub fn as_str(&self) -> &'static str {
        match self {
            MtModel::M2M100418M => "m2m100-418m",
            MtModel::Madlad3B => "madlad-3b",
            MtModel::Madlad10B => "madlad-10b",
            MtModel::HelsinkiEnEs => "helsinki-en-es",
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
