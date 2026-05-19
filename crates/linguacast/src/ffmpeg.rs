use anyhow::{anyhow, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Locate the ffmpeg binary. Prefers `LINGUACAST_FFMPEG`, then `PATH`.
pub fn locate() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("LINGUACAST_FFMPEG") {
        return Ok(PathBuf::from(p));
    }
    which::which("ffmpeg")
        .map_err(|e| anyhow!("ffmpeg not found on PATH (and LINGUACAST_FFMPEG not set): {e}"))
}

/// Extract a mono 16 kHz WAV — Whisper's preferred input format.
pub fn extract_audio_16k_mono(input: &Path, out_wav: &Path) -> Result<()> {
    let ffmpeg = locate()?;
    let status = Command::new(&ffmpeg)
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
        ])
        .arg(input)
        .args(["-vn", "-ac", "1", "-ar", "16000", "-f", "wav"])
        .arg(out_wav)
        .status()
        .with_context(|| format!("spawning ffmpeg ({})", ffmpeg.display()))?;
    if !status.success() {
        return Err(anyhow!("ffmpeg audio extraction failed ({status})"));
    }
    Ok(())
}

/// Mux a new audio track over the original video, dropping the original
/// audio. Re-encodes audio to AAC (mp4 container friendly) and copies video.
/// Preserves duration; if the new audio is shorter we let ffmpeg pad with
/// silence via `-shortest` on the *video* side instead — we want to keep the
/// full video and let trailing silence sit at the end.
pub fn mux_replace_audio(video: &Path, audio: &Path, out_mp4: &Path) -> Result<()> {
    let ffmpeg = locate()?;
    let status = Command::new(&ffmpeg)
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
        ])
        .arg(video)
        .arg("-i")
        .arg(audio)
        .args([
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
        ])
        .arg(out_mp4)
        .status()
        .with_context(|| format!("spawning ffmpeg ({})", ffmpeg.display()))?;
    if !status.success() {
        return Err(anyhow!("ffmpeg mux failed ({status})"));
    }
    Ok(())
}
