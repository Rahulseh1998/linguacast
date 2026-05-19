use crate::{
    cli::{self, AsrModel, Cli, MtModel, TtsSize},
    consent::{self, ConsentOutcome, ConsentRecord},
    device,
    ffmpeg,
    pack,
    progress::{humanize_error, PipelineProgress},
    sidecar::{self, ProgressEvent},
};
use anyhow::{anyhow, Context, Result};
use std::path::{Path, PathBuf};
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
    let cache_hint = cache_root_hint();
    eprintln!(
        "\nlinguacast pull — downloading model weights (~10 GB on first run)\n\
         Models land in {cache_hint}/\n\
         This is a one-time step; subsequent dubs use the warm cache.\n"
    );

    let mut sidecar = sidecar::Sidecar::launch(opts.python.as_deref(), &sidecar_dir)
        .map_err(humanize)?;
    sidecar.hello().map_err(humanize)?;

    let t = Instant::now();
    let (cache_root, models) = sidecar
        .pull(opts.asr.as_str(), opts.mt.as_str(), opts.tts_size.as_str())
        .map_err(humanize)?;
    info!(
        "pull complete in {:.1}s — cache at {}",
        t.elapsed().as_secs_f32(),
        cache_root
    );
    for (role, id) in &models {
        eprintln!("  ✓ {role}: {id}");
    }
    eprintln!("\nAll models cached. Run `linguacast <video> --langs es` to dub.");
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

    let mut sidecar = sidecar::Sidecar::launch(cli.python.as_deref(), &sidecar_dir)
        .map_err(humanize)?;
    sidecar.hello().map_err(humanize)?;

    // Transparent auto-pull: if the cache directory looks empty, prime it
    // before any stage opens its mouth. Lets users skip the explicit
    // `linguacast pull` (OPE-44 UX polish).
    maybe_auto_pull(&mut sidecar, asr, mt, tts_size)?;

    let work = tempfile::tempdir().context("creating temp working dir")?;
    let audio_wav = work.path().join("input-16k-mono.wav");

    let t = Instant::now();
    ffmpeg::extract_audio_16k_mono(input, &audio_wav).map_err(humanize)?;
    let duration_sec = ffmpeg::probe_duration_sec(input).unwrap_or(0.0);
    info!(
        "ffmpeg extract audio: {:.2}s · clip duration {:.1}s",
        t.elapsed().as_secs_f32(),
        duration_sec
    );

    // OPE-12 consent gate. Runs before any TTS render and before any
    // output is produced. The hash is computed over the 16k mono WAV the
    // TTS step will actually consume — renaming the source video does not
    // bypass it.
    let consent_record = run_consent_gate(&cli, &audio_wav)?;

    let pp = PipelineProgress::new(true);
    let mut outputs: Vec<PathBuf> = Vec::with_capacity(cli.langs.len());
    let mut failed: Vec<(String, String)> = Vec::new();

    for lang in &cli.langs {
        let mut bar = pp.lang_bar(&lang.0);
        match process_one_lang(
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
            &consent_record,
            &mut bar,
        ) {
            Ok(out_mp4) => {
                bar.finish_ok(&format!("→ {}", out_mp4.display()));
                outputs.push(out_mp4);
            }
            Err(err) => {
                let human = humanize_error(&format!("{err:#}"));
                bar.finish_err(&human);
                failed.push((lang.0.clone(), human));
            }
        }
    }

    if !failed.is_empty() {
        eprintln!("\n{} language(s) failed:", failed.len());
        for (l, e) in &failed {
            eprintln!("  - {l}: {e}");
        }
    }

    if cli.pack && !outputs.is_empty() {
        let input = cli.input.as_ref().expect("validated above");
        let stem = input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("input");
        let pack_path = cli.out_dir.join(format!("{stem}.pack.zip"));
        match pack::build_pack(input, &outputs, &pack_path) {
            Ok(()) => eprintln!("\nPack: {}", pack_path.display()),
            Err(err) => eprintln!(
                "\nPack build failed: {}",
                humanize_error(&format!("{err:#}"))
            ),
        }
    }

    if !failed.is_empty() && outputs.is_empty() {
        return Err(anyhow!(
            "all {} language(s) failed — see errors above",
            failed.len()
        ));
    }

    Ok(())
}

fn run_consent_gate(cli: &Cli, audio_wav: &std::path::Path) -> Result<ConsentRecord> {
    let store_dir = match &cli.consent_store_dir {
        Some(p) => p.clone(),
        None => consent::default_consent_dir()?,
    };
    let opts = consent::ConsentOpts {
        reference_audio: audio_wav,
        speaker_name: cli.speaker_name.as_deref(),
        consent_file: cli.i_have_speaker_consent.as_deref(),
        consent_store_dir: store_dir,
        refusal_list_override: cli.refusal_list.as_deref(),
        force_non_interactive: false,
    };
    let outcome = consent::obtain_consent(&opts)?;
    let rec = match &outcome {
        ConsentOutcome::Reused(r) => {
            info!(
                "voice-clone consent reused (hash={} signed_by={} ts={})",
                short_hash(&r.audio_sha256),
                r.signed_by,
                r.timestamp_iso
            );
            r.clone()
        }
        ConsentOutcome::NewlySigned(r) => {
            info!(
                "voice-clone consent recorded (hash={} mode={:?} signed_by={})",
                short_hash(&r.audio_sha256),
                r.mode,
                r.signed_by
            );
            r.clone()
        }
    };
    Ok(rec)
}

fn short_hash(h: &str) -> &str {
    if h.len() > 12 {
        &h[..12]
    } else {
        h
    }
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
    consent: &ConsentRecord,
    bar: &mut crate::progress::LangProgress,
) -> Result<PathBuf> {
    let dubbed_audio = {
        let mut p = std::env::temp_dir();
        p.push(format!("linguacast-dubbed-{}.wav", lang.0));
        p
    };

    let t = Instant::now();
    let mut current_stage: Option<String> = None;
    let mut on_progress = |ev: ProgressEvent| {
        if current_stage.as_deref() != Some(ev.stage.as_str()) {
            if let Some(prev) = current_stage.take() {
                bar.stage_done(&prev, 0.0);
            }
            bar.stage_start(&ev.stage);
            current_stage = Some(ev.stage.clone());
        }
        if let (Some(cur), Some(tot)) = (ev.current, ev.total) {
            bar.stage_substep(cur, tot, &ev.phase);
        }
    };

    let watermark_id_hex = watermark_id_from_consent(consent);
    let report = sidecar
        .run_dub(
            audio_wav,
            audio_wav, // source audio doubles as speaker reference
            &lang.0,
            &dubbed_audio,
            duration_sec,
            asr,
            mt,
            tts_size,
            device,
            Some(&watermark_id_hex),
            &mut on_progress,
        )
        .map_err(humanize)?;
    if let Some(prev) = current_stage.take() {
        bar.stage_done(&prev, 0.0);
    }

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

    bar.stage_start("mux");
    let t = Instant::now();
    let meta = build_consent_metadata(consent, &lang.0, &watermark_id_hex);
    ffmpeg::mux_replace_audio(input, &dubbed_audio, &out_mp4, &meta).map_err(humanize)?;
    bar.stage_done("mux", t.elapsed().as_secs_f64());
    info!(
        "ffmpeg mux ({}): {:.2}s → {}",
        lang.0,
        t.elapsed().as_secs_f32(),
        out_mp4.display()
    );

    Ok(out_mp4)
}

/// OPE-13: derive a deterministic 32-bit watermark id from the consent hash.
/// The high 32 bits of SHA-256(consent_hash_hex) — so any verifier seeing the
/// watermark can cross-check it against the consent hash carried in metadata,
/// and replacing the consent hash in metadata while keeping the audio exposes
/// the mismatch.
fn watermark_id_from_consent(rec: &ConsentRecord) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(rec.audio_sha256.as_bytes());
    let digest = h.finalize();
    let mut id = [0u8; 4];
    id.copy_from_slice(&digest[..4]);
    hex_u32(u32::from_be_bytes(id))
}

fn hex_u32(v: u32) -> String {
    format!("{v:08x}")
}

fn build_consent_metadata(
    rec: &ConsentRecord,
    target_lang: &str,
    watermark_id_hex: &str,
) -> Vec<(String, String)> {
    // MP4 container metadata. ffmpeg writes -metadata key=value into the
    // udta box; standard QuickTime keys (`comment`, `description`) are
    // surfaced by every player that reads MP4 metadata, custom keys land
    // as freeform atoms which ffprobe shows but most players ignore.
    // We mirror the consent hash into both `comment` (universal) and a
    // namespaced `linguacast_consent_hash` key (machine-parseable).
    let signer = &rec.signed_by;
    let comment = format!(
        "linguacast:consent_hash={hash};consent_ts={ts};signer={signer};lang={lang};version={ver};watermark_id={wid}",
        hash = rec.audio_sha256,
        ts = rec.timestamp_iso,
        lang = target_lang,
        ver = rec.linguacast_version,
        wid = watermark_id_hex,
    );
    vec![
        ("comment".to_string(), comment),
        ("linguacast_consent_hash".to_string(), rec.audio_sha256.clone()),
        (
            "linguacast_consent_timestamp".to_string(),
            rec.timestamp_iso.clone(),
        ),
        ("linguacast_consent_signer".to_string(), signer.clone()),
        ("linguacast_version".to_string(), rec.linguacast_version.clone()),
        // OPE-13: pair the audio watermark with a machine-parseable id in
        // metadata. Stripping `-map_metadata -1` removes this; the watermark
        // bits survive AAC re-encode (see docs/watermark.md).
        ("linguacast_watermark_id".to_string(), watermark_id_hex.to_string()),
        (
            "linguacast_watermark_algo".to_string(),
            "patchwork-spread-spectrum-v1".to_string(),
        ),
    ]
}

// ---- Verify --------------------------------------------------------------

pub struct VerifyOpts {
    pub input: PathBuf,
    pub python: Option<PathBuf>,
    pub json: bool,
}

pub fn run_verify(opts: VerifyOpts) -> Result<()> {
    if !opts.input.exists() {
        return Err(anyhow!("input file not found: {}", opts.input.display()));
    }
    let sidecar_dir = locate_sidecar_dir()?;
    let mut sidecar = sidecar::Sidecar::launch(opts.python.as_deref(), &sidecar_dir)
        .map_err(humanize)?;
    sidecar.hello().map_err(humanize)?;

    // Extract a 24 kHz mono WAV — the rate the watermark embedder wrote at,
    // so the detector sees identical frequency-bin mapping.
    let work = tempfile::tempdir().context("creating verify work dir")?;
    let candidate_wav = work.path().join("candidate-24k-mono.wav");
    ffmpeg::extract_audio_24k_mono(&opts.input, &candidate_wav).map_err(humanize)?;

    // Container metadata (may be stripped — the watermark is load-bearing,
    // metadata is convenience). Probe is best-effort: a verifier that can't
    // read tags still emits a meaningful report.
    let meta = ffmpeg::probe_metadata(&opts.input).unwrap_or(serde_json::json!({}));
    let format_tags = meta
        .get("format")
        .and_then(|f| f.get("tags"))
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    let report = sidecar.verify(&candidate_wav).map_err(humanize)?;

    let meta_wm_id = format_tags
        .get("linguacast_watermark_id")
        .and_then(|v| v.as_str())
        .map(str::to_lowercase);
    let audio_wm_id = report.watermark_id.as_deref().map(str::to_lowercase);
    let metadata_audio_match = match (&meta_wm_id, &audio_wm_id) {
        (Some(m), Some(a)) => Some(m == a),
        _ => None,
    };

    if opts.json {
        let v = serde_json::json!({
            "input": opts.input.display().to_string(),
            "watermark": {
                "detected": report.detected,
                "confidence": report.confidence,
                "watermark_id": report.watermark_id,
                "version": report.version,
                "crc_ok": report.crc_ok,
                "repeats_voted": report.repeats_voted,
                "bit_offset_frames": report.bit_offset_frames,
                "sample_rate": report.sample_rate,
                "duration_sec": report.duration_sec,
                "elapsed_sec": report.elapsed_sec,
                "algorithm": report.algorithm,
            },
            "metadata": format_tags,
            "metadata_audio_match": metadata_audio_match,
        });
        println!("{}", serde_json::to_string_pretty(&v)?);
    } else {
        print_verify_human(&opts.input, &report, &format_tags, metadata_audio_match);
    }

    Ok(())
}

fn print_verify_human(
    input: &Path,
    r: &sidecar::VerifyReport,
    meta_tags: &serde_json::Value,
    metadata_audio_match: Option<bool>,
) {
    println!("LinguaCast verify — {}", input.display());
    println!();
    println!("Audio watermark");
    println!(
        "  detected      : {}",
        if r.detected { "yes" } else { "no" }
    );
    println!("  confidence    : {:.3}", r.confidence);
    if let Some(id) = &r.watermark_id {
        println!("  watermark id  : {id}");
    }
    if let Some(v) = r.version {
        println!("  version byte  : 0x{:02x}", v as u8);
    }
    if let Some(ok) = r.crc_ok {
        println!("  crc           : {}", if ok { "ok" } else { "BAD" });
    }
    println!("  repeats voted : {}", r.repeats_voted);
    println!("  sample rate   : {} Hz", r.sample_rate);
    println!("  audio duration: {:.2} s", r.duration_sec);
    println!("  detect time   : {:.3} s", r.elapsed_sec);
    println!("  algorithm     : {}", r.algorithm);

    println!();
    println!("Container metadata (ID3/XMP-style, may be stripped)");
    if let Some(map) = meta_tags.as_object() {
        if map.is_empty() {
            println!("  (no metadata tags — may have been stripped)");
        } else {
            for (k, v) in map {
                let s = match v {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                println!("  {k:30} : {s}");
            }
        }
    } else {
        println!("  (no metadata block)");
    }

    println!();
    match metadata_audio_match {
        Some(true) => println!(
            "Provenance: ✓ audio watermark id matches the linguacast_watermark_id in metadata."
        ),
        Some(false) => println!(
            "Provenance: ✗ MISMATCH — the watermark id recovered from audio differs from \
             the linguacast_watermark_id claimed in metadata. The metadata may have been \
             tampered with, or the audio was re-muxed under a different LinguaCast output's tags."
        ),
        None => {
            if r.detected {
                println!(
                    "Provenance: watermark detected, metadata id missing (likely stripped by \
                     `ffmpeg -map_metadata -1`). The audio bit-payload still identifies this as \
                     LinguaCast-generated."
                );
            } else {
                println!(
                    "Provenance: no LinguaCast watermark detected. This is either not a \
                     LinguaCast output, or the audio has been heavily re-processed."
                );
            }
        }
    }
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

fn cache_dir() -> PathBuf {
    if let Ok(p) = std::env::var("LINGUACAST_CACHE_DIR") {
        return PathBuf::from(p);
    }
    directories::BaseDirs::new()
        .map(|b| b.cache_dir().join("linguacast"))
        .unwrap_or_else(|| PathBuf::from(".cache/linguacast"))
}

fn cache_root_hint() -> String {
    cache_dir().display().to_string()
}

fn cache_looks_empty(cache: &Path) -> bool {
    if !cache.exists() {
        return true;
    }
    let hf = cache.join("hf");
    let whisper = cache.join("models").join("whisper");
    let whisper_legacy = cache.join("whisper");
    !(hf.exists() || whisper.exists() || whisper_legacy.exists())
}

fn maybe_auto_pull(
    sidecar: &mut sidecar::Sidecar,
    asr: &str,
    mt: &str,
    tts_size: &str,
) -> Result<()> {
    let cache = cache_dir();
    if !cache_looks_empty(&cache) {
        return Ok(());
    }
    eprintln!(
        "\nlinguacast: first run — pulling model weights to {} (~10 GB).\n\
         This takes 10–20 minutes on a fast connection. Subsequent dubs reuse the cache.\n",
        cache.display()
    );
    let t = Instant::now();
    let (_root, models) = sidecar.pull(asr, mt, tts_size).map_err(humanize)?;
    info!(
        "auto-pull complete in {:.1}s ({} models)",
        t.elapsed().as_secs_f32(),
        models.len()
    );
    for (role, id) in &models {
        eprintln!("  ✓ {role}: {id}");
    }
    eprintln!();
    Ok(())
}

fn humanize(err: anyhow::Error) -> anyhow::Error {
    let raw = format!("{err:#}");
    let mapped = humanize_error(&raw);
    if mapped == raw {
        err
    } else {
        anyhow!(mapped)
    }
}
