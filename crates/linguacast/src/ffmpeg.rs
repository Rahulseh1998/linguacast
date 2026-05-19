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

/// Locate ffprobe (ships with ffmpeg). Same precedence as locate().
pub fn locate_ffprobe() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("LINGUACAST_FFPROBE") {
        return Ok(PathBuf::from(p));
    }
    which::which("ffprobe")
        .map_err(|e| anyhow!("ffprobe not found on PATH (and LINGUACAST_FFPROBE not set): {e}"))
}

/// Probe duration in seconds via ffprobe. The TTS stage needs this to size
/// the output track; we ask ffprobe rather than re-decoding because it's
/// cheap and avoids spinning up another ffmpeg pass.
pub fn probe_duration_sec(input: &Path) -> Result<f32> {
    let ffprobe = locate_ffprobe()?;
    let output = Command::new(&ffprobe)
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(input)
        .output()
        .with_context(|| format!("spawning ffprobe ({})", ffprobe.display()))?;
    if !output.status.success() {
        return Err(anyhow!(
            "ffprobe duration probe failed ({status}): {stderr}",
            status = output.status,
            stderr = String::from_utf8_lossy(&output.stderr)
        ));
    }
    let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
    s.parse::<f32>().map_err(|e| {
        anyhow!("could not parse ffprobe duration {s:?}: {e}")
    })
}

/// Probe container-level metadata as a flat JSON object. The verifier
/// (OPE-13) reads the `linguacast_consent_hash` / `linguacast_watermark_id`
/// keys back out of MP4 udta atoms via this path.
pub fn probe_metadata(input: &Path) -> Result<serde_json::Value> {
    let ffprobe = locate_ffprobe()?;
    let output = Command::new(&ffprobe)
        .args([
            "-v", "error",
            "-show_entries", "format_tags:stream_tags",
            "-of", "json",
        ])
        .arg(input)
        .output()
        .with_context(|| format!("spawning ffprobe ({})", ffprobe.display()))?;
    if !output.status.success() {
        return Err(anyhow!(
            "ffprobe metadata probe failed ({status}): {stderr}",
            status = output.status,
            stderr = String::from_utf8_lossy(&output.stderr)
        ));
    }
    let value: serde_json::Value = serde_json::from_slice(&output.stdout)
        .with_context(|| format!("parsing ffprobe metadata json ({}b)", output.stdout.len()))?;
    Ok(value)
}

/// Extract a mono 16 kHz WAV — Whisper's preferred input format.
pub fn extract_audio_16k_mono(input: &Path, out_wav: &Path) -> Result<()> {
    extract_audio_mono(input, out_wav, 16_000)
}

/// Extract a mono 24 kHz WAV — Qwen3-TTS sample-rate; matches the rate the
/// watermark embedder writes at, so the OPE-13 detector sees the same bins.
pub fn extract_audio_24k_mono(input: &Path, out_wav: &Path) -> Result<()> {
    extract_audio_mono(input, out_wav, 24_000)
}

fn extract_audio_mono(input: &Path, out_wav: &Path, sr: u32) -> Result<()> {
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
        .arg("-vn")
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg(sr.to_string())
        .arg("-f")
        .arg("wav")
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
///
/// `metadata` writes container-level `-metadata key=value` pairs (mp4 udta).
/// Used for the OPE-12 consent provenance fields.
pub fn mux_replace_audio(
    video: &Path,
    audio: &Path,
    out_mp4: &Path,
    metadata: &[(String, String)],
) -> Result<()> {
    let ffmpeg = locate()?;
    let mut cmd = Command::new(&ffmpeg);
    cmd.args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(video)
        .arg("-i")
        .arg(audio)
        .args([
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            // +use_metadata_tags is required for the mp4 muxer to preserve
            // namespaced keys (e.g. linguacast_consent_hash) as freeform
            // udta atoms. Without it ffmpeg drops everything outside its
            // well-known QuickTime tag list.
            "-movflags", "+faststart+use_metadata_tags",
        ]);
    for (k, v) in metadata {
        cmd.arg("-metadata").arg(format!("{k}={v}"));
    }
    let status = cmd
        .arg(out_mp4)
        .status()
        .with_context(|| format!("spawning ffmpeg ({})", ffmpeg.display()))?;
    if !status.success() {
        return Err(anyhow!("ffmpeg mux failed ({status})"));
    }
    Ok(())
}
