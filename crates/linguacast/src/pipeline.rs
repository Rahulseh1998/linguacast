use crate::{
    cli::{self, AsrModel, Cli, MtModel, TtsSize},
    device,
    ffmpeg, sidecar,
};
use anyhow::{anyhow, Context, Result};
use std::path::PathBuf;
use std::time::Instant;
use tracing::info;

// ---- Pull ----------------------------------------------------------------

pub struct PullOpts {
    pub asr: AsrModel,
    pub mt: MtModel,
    pub tts_size: TtsSize,
    pub python: Option<PathBuf>,
}

pub fn run_pull(opts: PullOpts) -> Result<()> {
    let sidecar_dir = locate_sidecar_dir()?;
    info!("sidecar dir: {}", sidecar_dir.display());

    info!(
        "pulling models: asr={} mt={} tts={}",
        opts.asr.as_str(),
        opts.mt.as_str(),
        opts.tts_size.as_str()
    );
    let cache_hint = directories::BaseDirs::new()
        .map(|b| b.cache_dir().join("linguacast").display().to_string())
        .unwrap_or_else(|| "~/.cache/linguacast".to_string());
    eprintln!(
        "\nlinguacast pull — downloading model weights (~10 GB on first run)\n\
         Models land in {cache_hint}/\n\
         This is a one-time step; subsequent dubs use the warm cache.\n"
    );

    let mut sidecar = sidecar::Sidecar::launch(opts.python.as_deref(), &sidecar_dir)?;
    sidecar.hello()?;

    let t = Instant::now();
    let (cache_root, models) = sidecar.pull(
        opts.asr.as_str(),
        opts.mt.as_str(),
        opts.tts_size.as_str(),
    )?;
    info!(
        "pull complete in {:.1}s — cache at {}",
        t.elapsed().as_secs_f32(),
        cache_root
    );
    for (role, id) in &models {
        eprintln!("  ✓ {role}: {id}");
    }
    eprintln!("\nAll models cached. Run `linguacast dub <video> --langs es` to dub.");
    Ok(())
}

// ---- Dub -----------------------------------------------------------------

pub fn run_dub(cli: Cli) -> Result<()> {
    let input = cli
        .input
        .as_ref()
        .ok_or_else(|| anyhow!("input file required (or run `linguacast pull` to download models)"))?;
    if !input.exists() {
        return Err(anyhow!("input file not found: {}", input.display()));
    }
    if cli.langs.is_empty() {
        return Err(anyhow!("--langs must include at least one language"));
    }
    std::fs::create_dir_all(&cli.out_dir).with_context(|| {
        format!("creating output directory {}", cli.out_dir.display())
    })?;

    let device = device::resolve(cli.device.as_ref());
    info!("device: {}", device.as_str());

    let tts_engine = pick_tts_engine(&cli);
    info!("tts engine: {tts_engine}");

    let asr = cli.asr.as_str();
    let mt = cli.mt.as_str();
    let tts_size = cli.tts_size.as_str();
    info!("models: asr={asr} mt={mt} tts={tts_size}");

    let sidecar_dir = locate_sidecar_dir()?;
    info!("sidecar dir: {}", sidecar_dir.display());

    let mut sidecar = sidecar::Sidecar::launch(cli.python.as_deref(), &sidecar_dir)?;
    sidecar.hello()?;

    let work = tempfile::tempdir().context("creating temp working dir")?;
    let audio_wav = work.path().join("input-16k-mono.wav");

    // Extract audio once; the WAV serves as both the Whisper input and the
    // TTS speaker reference clip for voice cloning.
    let t = Instant::now();
    ffmpeg::extract_audio_16k_mono(input, &audio_wav)?;
    let duration_sec = ffmpeg::probe_duration_sec(input).unwrap_or(0.0);
    info!(
        "ffmpeg extract audio: {:.2}s · clip duration {:.1}s",
        t.elapsed().as_secs_f32(),
        duration_sec
    );

    for lang in &cli.langs {
        process_one_lang(
            &cli,
            &device,
            tts_engine,
            &mut sidecar,
            &audio_wav,
            duration_sec,
            asr,
            mt,
            tts_size,
            lang,
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn process_one_lang(
    cli: &Cli,
    device: &device::Device,
    _tts_engine: &str,
    sidecar: &mut sidecar::Sidecar,
    audio_wav: &std::path::Path,
    duration_sec: f32,
    asr: &str,
    mt: &str,
    tts_size: &str,
    lang: &cli::Lang,
) -> Result<()> {
    if !cli.i_understand_voice_clone_risks {
        return Err(anyhow!(
            "voice clone consent required: re-run with --i-understand-voice-clone-risks. \
             The real consent gate (OPE-12) lands in week 3 and will become the default check."
        ));
    }

    let dubbed_audio = {
        let mut p = std::env::temp_dir();
        p.push(format!("linguacast-dubbed-{}.wav", lang.0));
        p
    };

    let t = Instant::now();
    let report = sidecar.run_dub(
        audio_wav,
        audio_wav, // source audio doubles as speaker reference
        &lang.0,
        &dubbed_audio,
        duration_sec,
        asr,
        mt,
        tts_size,
        device,
    )?;

    // Log per-stage RSS for the 8 GB-fit conversation.
    info!(
        "run_dub ({} → {}): {:.1}s audio · {} segments · {:.1}s wall · peak RSS {:.0} MB",
        report.language,
        report.target_lang,
        report.duration_sec,
        report.segments_rendered,
        t.elapsed().as_secs_f32(),
        report.peak_rss_mb,
    );
    for stage in &report.stages {
        info!(
            "  stage {} ({}): {:.1}s · peak RSS {:.0} MB",
            stage.name, stage.model, stage.stage_seconds, stage.peak_rss_mb
        );
    }

    let input = cli.input.as_ref().expect("validated above");
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("input");
    let out_mp4 = cli.out_dir.join(format!("{stem}.{}.mp4", lang.0));

    let t = Instant::now();
    ffmpeg::mux_replace_audio(input, &dubbed_audio, &out_mp4)?;
    info!(
        "ffmpeg mux ({}): {:.2}s → {}",
        lang.0,
        t.elapsed().as_secs_f32(),
        out_mp4.display()
    );
    eprintln!("\nOutput: {}", out_mp4.display());

    Ok(())
}

fn pick_tts_engine(cli: &Cli) -> &'static str {
    if let Some(explicit) = &cli.tts {
        return match explicit {
            cli::TtsEngine::Qwen3Tts => "qwen3-tts",
            cli::TtsEngine::Qwen3TtsQuantized => "qwen3-tts-quantized",
            cli::TtsEngine::Voicebox => "voicebox",
        };
    }
    "qwen3-tts"
}

fn locate_sidecar_dir() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("LINGUACAST_SIDECAR_DIR") {
        let p = PathBuf::from(p);
        if p.is_dir() {
            return Ok(p);
        }
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(mut dir) = exe.parent().map(std::path::Path::to_path_buf) {
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
