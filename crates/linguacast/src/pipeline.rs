use crate::{
    cli::{self, Args, Lang},
    device::{self, Device},
    ffmpeg, sidecar,
};
use anyhow::{anyhow, Context, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::info;

pub fn run(args: Args) -> Result<()> {
    if !args.input.exists() {
        return Err(anyhow!("input file not found: {}", args.input.display()));
    }
    if args.langs.is_empty() {
        return Err(anyhow!("--langs must include at least one language"));
    }
    std::fs::create_dir_all(&args.out_dir).with_context(|| {
        format!("creating output directory {}", args.out_dir.display())
    })?;

    let device = device::resolve(args.device.as_ref());
    info!("device: {}", device.as_str());

    let engine = pick_engine(&args, &device);
    info!("tts engine: {engine}");

    // The sidecar directory lives next to the binary in a dev checkout
    // (`sidecar/` at the repo root) and inside the bundle for shipped builds.
    let sidecar_dir = locate_sidecar_dir()?;
    info!("sidecar dir: {}", sidecar_dir.display());

    let mut sidecar = sidecar::Sidecar::launch(args.python.as_deref(), &sidecar_dir)?;
    sidecar.hello()?;

    let work = tempfile::tempdir().context("creating temp working dir")?;
    let audio_wav = work.path().join("input-16k-mono.wav");

    let t = Instant::now();
    ffmpeg::extract_audio_16k_mono(&args.input, &audio_wav)?;
    info!("ffmpeg extract audio: {:.2}s", t.elapsed().as_secs_f32());

    let t = Instant::now();
    let (source_lang, transcript) = sidecar.transcribe(&audio_wav, &device)?;
    info!(
        "whisper: detected language {} · {} segments · {:.2}s",
        source_lang,
        transcript.len(),
        t.elapsed().as_secs_f32()
    );

    for lang in &args.langs {
        process_one_lang(
            &args,
            &device,
            engine,
            &mut sidecar,
            &source_lang,
            &transcript,
            &audio_wav,
            work.path(),
            lang,
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn process_one_lang(
    args: &Args,
    device: &Device,
    engine: &str,
    sidecar: &mut sidecar::Sidecar,
    source_lang: &str,
    transcript: &[sidecar::Segment],
    reference_audio: &Path,
    work_dir: &Path,
    lang: &Lang,
) -> Result<()> {
    if !args.i_understand_voice_clone_risks {
        // Week-1: placeholder consent gate. The real implementation
        // lands in week 3 (OPE-12); for now we emit a loud warning rather
        // than block, but we do refuse without the explicit flag. That keeps
        // the spike honest about what's still missing.
        return Err(anyhow!(
            "voice clone consent required: re-run with --i-understand-voice-clone-risks. \
             The real consent gate (OPE-12) lands in week 3 and will become the default check."
        ));
    }

    let t = Instant::now();
    let translated = sidecar.translate(transcript, source_lang, &lang.0, device)?;
    info!(
        "madlad-400: {} → {} · {} segments · {:.2}s",
        source_lang,
        lang.0,
        translated.len(),
        t.elapsed().as_secs_f32()
    );

    let dubbed_audio = work_dir.join(format!("dubbed-{}.wav", lang.0));
    let t = Instant::now();
    let (out_audio, duration) = sidecar.tts(
        &translated,
        reference_audio,
        &lang.0,
        &dubbed_audio,
        device,
        engine,
    )?;
    info!(
        "tts ({engine}): {:.2}s audio rendered in {:.2}s",
        duration,
        t.elapsed().as_secs_f32()
    );

    let stem = args
        .input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("input");
    let out_mp4 = args.out_dir.join(format!("{stem}.{}.mp4", lang.0));
    let t = Instant::now();
    ffmpeg::mux_replace_audio(&args.input, &out_audio, &out_mp4)?;
    info!(
        "ffmpeg mux ({}): {:.2}s · → {}",
        lang.0,
        t.elapsed().as_secs_f32(),
        out_mp4.display()
    );

    Ok(())
}

fn pick_engine(args: &Args, _device: &Device) -> &'static str {
    if let Some(explicit) = &args.tts {
        return match explicit {
            cli::TtsEngine::Qwen3Tts => "qwen3-tts",
            cli::TtsEngine::Qwen3TtsQuantized => "qwen3-tts-quantized",
            cli::TtsEngine::Voicebox => "voicebox",
        };
    }
    // Auto-select. Week-1 default: full-precision Qwen3-TTS on ≥12 GB, else
    // the quantized variant. This is the engine decision the spike validates.
    // We don't have an accurate cross-platform "free unified memory" probe
    // yet, so the default is conservative: quantized everywhere until the
    // benchmark on OPE-19 says otherwise.
    "qwen3-tts-quantized"
}

fn locate_sidecar_dir() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("LINGUACAST_SIDECAR_DIR") {
        let p = PathBuf::from(p);
        if p.is_dir() {
            return Ok(p);
        }
    }
    // Dev layout: <repo>/sidecar/ alongside crates/linguacast/.
    // Walk up from the current exe (release build) and from CARGO_MANIFEST_DIR (cargo run).
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(mut dir) = exe.parent().map(Path::to_path_buf) {
            for _ in 0..4 {
                candidates.push(dir.join("sidecar"));
                if !dir.pop() {
                    break;
                }
            }
        }
    }
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let mut dir = PathBuf::from(manifest);
        for _ in 0..3 {
            candidates.push(dir.join("sidecar"));
            if !dir.pop() {
                break;
            }
        }
    }
    for c in &candidates {
        if c.join("main.py").exists() {
            return Ok(c.clone());
        }
    }
    Err(anyhow!(
        "could not locate the Python sidecar directory. \
         Set LINGUACAST_SIDECAR_DIR to <repo>/sidecar."
    ))
}
