use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::time::Duration;

/// Stage labels in the order the pipeline runs them.
pub const STAGES: &[&str] = &["asr", "mt", "tts", "mux"];

/// Per-language progress: one bar that walks through asr → mt → tts → mux.
/// Each bar tracks 4 stage steps (length=4) so the user sees concrete advance
/// even when the underlying sidecar doesn't emit per-segment progress.
pub struct LangProgress {
    bar: ProgressBar,
    stage_idx: usize,
}

impl LangProgress {
    pub fn stage_start(&mut self, stage: &str) {
        self.stage_idx = STAGES.iter().position(|s| *s == stage).unwrap_or(0);
        self.bar
            .set_message(format!("{} · {}", self.bar.prefix(), stage));
        self.bar.set_position(self.stage_idx as u64);
        self.bar.enable_steady_tick(Duration::from_millis(120));
    }

    pub fn stage_substep(&self, current: u64, total: u64, label: &str) {
        // Render sub-step inside a stage as "stage 2/11" by squeezing into
        // the message; the bar's discrete length stays at 4 (one per stage).
        self.bar.set_message(format!(
            "{} · {label} {current}/{total}",
            self.bar.prefix(),
        ));
    }

    pub fn stage_done(&self, stage: &str, elapsed_sec: f64) {
        let idx = STAGES.iter().position(|s| *s == stage).unwrap_or(0);
        self.bar.set_position((idx + 1) as u64);
        self.bar.set_message(format!(
            "{} · {} ✓ {:.1}s",
            self.bar.prefix(),
            stage,
            elapsed_sec
        ));
    }

    pub fn finish_ok(&self, summary: &str) {
        self.bar.set_position(STAGES.len() as u64);
        self.bar
            .finish_with_message(format!("{} · {summary}", self.bar.prefix()));
    }

    pub fn finish_err(&self, err: &str) {
        self.bar
            .finish_with_message(format!("{} · ✗ {err}", self.bar.prefix()));
    }
}

/// Owns the global MultiProgress and the per-language bars.
pub struct PipelineProgress {
    multi: MultiProgress,
}

impl PipelineProgress {
    pub fn new(enabled: bool) -> Self {
        let multi = MultiProgress::new();
        if !enabled {
            multi.set_draw_target(indicatif::ProgressDrawTarget::hidden());
        }
        Self { multi }
    }

    pub fn lang_bar(&self, lang: &str) -> LangProgress {
        let style = ProgressStyle::with_template(
            "{prefix:>4} [{bar:24.cyan/blue}] {msg}",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("█▉▊▋▌▍▎▏ ");
        let pb = self.multi.add(ProgressBar::new(STAGES.len() as u64));
        pb.set_style(style);
        pb.set_prefix(lang.to_string());
        pb.set_message("queued".to_string());
        LangProgress { bar: pb, stage_idx: 0 }
    }

    pub fn println(&self, line: impl AsRef<str>) {
        let _ = self.multi.println(line);
    }
}

/// Map raw sidecar/HF/ffmpeg errors to one-line, human-readable messages
/// with a concrete next step. Falls through to the original message for
/// patterns we don't recognise — we'd rather be slightly noisier than
/// silently hide an unexpected stack trace.
pub fn humanize_error(raw: &str) -> String {
    let lower = raw.to_ascii_lowercase();

    if lower.contains("repositorynotfounderror") || lower.contains("repository not found") {
        return format!(
            "Hugging Face model not found — check internet access, or run `linguacast pull` once you're online. ({raw})"
        );
    }
    if lower.contains("connectionerror")
        || lower.contains("connection error")
        || lower.contains("max retries exceeded")
        || lower.contains("nameresolutionerror")
        || lower.contains("temporary failure in name resolution")
        || lower.contains("could not connect to huggingface.co")
    {
        return format!(
            "Could not reach Hugging Face — check your internet, then re-run. Model weights live in ~/.cache/linguacast/."
        );
    }
    if lower.contains("disk quota exceeded") || lower.contains("no space left on device") {
        return "Out of disk space while downloading model weights (~10 GB needed). Free space in $HOME and retry.".into();
    }
    if lower.contains("mps")
        && (lower.contains("out of memory")
            || lower.contains("oom")
            || lower.contains("watermark")
            || lower.contains("placeholder storage has not been allocated"))
    {
        return "Apple Silicon MPS ran out of unified memory. Re-run with `--tts-size 0.6B` (smaller TTS) and/or `--asr medium`, or close memory-heavy apps.".into();
    }
    if lower.contains("cuda")
        && (lower.contains("out of memory") || lower.contains("cuda error"))
    {
        return "CUDA ran out of memory. Re-run with `--tts-size 0.6B` and/or `--asr medium`, or use `--device cpu`.".into();
    }
    if lower.contains("ffmpeg") && lower.contains("not found") {
        return "ffmpeg is not on $PATH. Install it (`brew install ffmpeg` / `apt install ffmpeg`) and re-run.".into();
    }
    if lower.contains("ffmpeg")
        && (lower.contains("muxer") || lower.contains("invalid data found") || lower.contains("does not contain any stream"))
    {
        return format!("ffmpeg could not mux the output — the input video may be unreadable. ({raw})");
    }
    if lower.contains("qwen-tts package is required") || lower.contains("no module named 'qwen_tts'") {
        return "qwen-tts is missing. Install the sidecar deps: `cd sidecar && .venv/bin/pip install -r requirements.txt`.".into();
    }
    if lower.contains("no module named 'faster_whisper'") {
        return "faster-whisper is missing. Install the sidecar deps: `cd sidecar && .venv/bin/pip install -r requirements.txt`.".into();
    }
    if lower.contains("no module named 'transformers'") {
        return "transformers is missing. Install the sidecar deps: `cd sidecar && .venv/bin/pip install -r requirements.txt`.".into();
    }
    if lower.contains("voice clone consent required") {
        // Already human-readable.
        return raw.to_string();
    }
    raw.to_string()
}
