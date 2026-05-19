//! `--pack` output mode (OPE-44).
//!
//! After all language outputs render, bundle them into a single
//! shareable zip with:
//!   * the per-language MP4s
//!   * a 16:9 PNG thumbnail per language (grabbed at the 50% mark)
//!   * a contact-sheet GIF cycling all outputs (1 fps, 480p)
//!   * a manifest.json listing the contents
//!
//! Two shellouts to ffmpeg: one per thumbnail (cheap) and one for the GIF
//! (palette-aware). Then a single zip writer. No new dependencies beyond
//! the `zip` crate already declared in Cargo.toml.

use anyhow::{anyhow, Context, Result};
use serde::Serialize;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::info;
use zip::write::SimpleFileOptions;

use crate::ffmpeg;

#[derive(Serialize)]
struct PackManifest<'a> {
    schema: u32,
    source_video: &'a str,
    languages: Vec<PackLang<'a>>,
    contact_sheet_gif: &'a str,
}

#[derive(Serialize)]
struct PackLang<'a> {
    lang: &'a str,
    mp4: &'a str,
    thumbnail: &'a str,
}

/// Build a `<stem>.pack.zip` containing all outputs + thumbnails + a
/// contact-sheet GIF. Writes the artefacts into a temp dir then streams
/// them into the zip in deflate mode.
pub fn build_pack(source: &Path, outputs: &[PathBuf], pack_path: &Path) -> Result<()> {
    if outputs.is_empty() {
        return Err(anyhow!("no language outputs to pack"));
    }
    let work = tempfile::tempdir().context("creating pack working directory")?;
    let work_dir = work.path();

    // Per-language thumbnails.
    let mut thumbs: Vec<(String, PathBuf)> = Vec::with_capacity(outputs.len());
    for mp4 in outputs {
        let lang = extract_lang_tag(mp4).unwrap_or_else(|| "unknown".to_string());
        let thumb = work_dir.join(format!("thumb.{lang}.png"));
        extract_thumbnail(mp4, &thumb).with_context(|| {
            format!("thumbnail for {} ({})", lang, mp4.display())
        })?;
        thumbs.push((lang, thumb));
    }

    // Contact sheet GIF. 1 fps, 480-wide, cycling the inputs.
    let gif_path = work_dir.join("contact-sheet.gif");
    build_contact_sheet_gif(outputs, &gif_path).context("building contact-sheet GIF")?;

    // Manifest. Lives at the root of the zip.
    let manifest_path = work_dir.join("manifest.json");
    let manifest_entries: Vec<PackLang> = outputs
        .iter()
        .zip(&thumbs)
        .map(|(mp4, (lang, thumb))| PackLang {
            lang: lang.as_str(),
            mp4: file_name_str(mp4),
            thumbnail: file_name_str(thumb),
        })
        .collect();
    let manifest = PackManifest {
        schema: 1,
        source_video: file_name_str(source),
        languages: manifest_entries,
        contact_sheet_gif: file_name_str(&gif_path),
    };
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&manifest_path, manifest_json.as_bytes())?;

    // Zip everything.
    if let Some(parent) = pack_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("creating pack parent directory {}", parent.display())
            })?;
        }
    }
    let f = File::create(pack_path)
        .with_context(|| format!("creating pack file {}", pack_path.display()))?;
    let mut zip = zip::ZipWriter::new(f);
    let opts: SimpleFileOptions = SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated)
        .unix_permissions(0o644);

    add_to_zip(&mut zip, &manifest_path, opts)?;
    add_to_zip(&mut zip, &gif_path, opts)?;
    for mp4 in outputs {
        add_to_zip(&mut zip, mp4, opts)?;
    }
    for (_, thumb) in &thumbs {
        add_to_zip(&mut zip, thumb, opts)?;
    }
    zip.finish().context("finalising pack zip")?;
    info!(
        "pack: {} languages, contact-sheet GIF, manifest → {}",
        outputs.len(),
        pack_path.display()
    );
    Ok(())
}

fn add_to_zip<W: Write + std::io::Seek>(
    zip: &mut zip::ZipWriter<W>,
    path: &Path,
    opts: SimpleFileOptions,
) -> Result<()> {
    let name = file_name_str(path);
    zip.start_file(name, opts)
        .with_context(|| format!("zip start_file {name}"))?;
    let mut f = File::open(path)
        .with_context(|| format!("reading {} for zip", path.display()))?;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f
            .read(&mut buf)
            .with_context(|| format!("reading {} for zip", path.display()))?;
        if n == 0 {
            break;
        }
        zip.write_all(&buf[..n])
            .with_context(|| format!("writing {name} into pack"))?;
    }
    Ok(())
}

fn file_name_str(p: &Path) -> &str {
    p.file_name().and_then(|n| n.to_str()).unwrap_or("file")
}

/// Pull the `xx` out of `name.xx.mp4`.
fn extract_lang_tag(mp4: &Path) -> Option<String> {
    let stem = mp4.file_stem()?.to_str()?;
    stem.rsplit_once('.').map(|(_, lang)| lang.to_string())
}

fn extract_thumbnail(mp4: &Path, out_png: &Path) -> Result<()> {
    let ffmpeg = ffmpeg::locate()?;
    // Try to take the frame at 50% duration. Falling back to 1s if probe fails.
    let mid = ffmpeg::probe_duration_sec(mp4).unwrap_or(2.0) / 2.0;
    let mid = if mid.is_finite() && mid > 0.5 { mid } else { 1.0 };
    let status = Command::new(&ffmpeg)
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
        ])
        .arg(format!("{mid:.2}"))
        .arg("-i")
        .arg(mp4)
        .args([
            "-frames:v",
            "1",
            "-vf",
            // 16:9 thumbnail: shrink to fit, then pad with black bars so
            // even portrait or small clips end up at 1280x720.
            "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:black,setsar=1",
        ])
        .arg(out_png)
        .status()
        .with_context(|| format!("spawning ffmpeg for thumbnail ({})", ffmpeg.display()))?;
    if !status.success() {
        return Err(anyhow!("ffmpeg thumbnail extraction failed ({status})"));
    }
    Ok(())
}

/// Cycle each output for 1 second at 480 wide, output a single GIF.
/// Uses a one-shot palettegen+paletteuse filter graph so colours don't
/// rot. The pre-existing palette generator gets fed the concat-filter
/// stream so it sees all 12 outputs at once.
fn build_contact_sheet_gif(outputs: &[PathBuf], out_gif: &Path) -> Result<()> {
    let ffmpeg = ffmpeg::locate()?;
    let n = outputs.len();
    if n == 0 {
        return Err(anyhow!("no outputs to build contact sheet from"));
    }
    let mut cmd = Command::new(&ffmpeg);
    cmd.args(["-y", "-hide_banner", "-loglevel", "error"]);

    // 1s per video, snip start. -ss 1 -t 1 keeps things fast.
    for mp4 in outputs {
        cmd.arg("-ss").arg("1").arg("-t").arg("1").arg("-i").arg(mp4);
    }

    // Filter graph: scale each input to 480:-2, concat them, then split off
    // a palettegen + paletteuse pair.
    let mut filter = String::new();
    for i in 0..n {
        filter.push_str(&format!(
            "[{i}:v]scale=480:-2:flags=lanczos,setsar=1,fps=12[v{i}];"
        ));
    }
    for i in 0..n {
        filter.push_str(&format!("[v{i}]"));
    }
    filter.push_str(&format!(
        "concat=n={n}:v=1:a=0[concat];[concat]split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    ));

    let status = cmd
        .arg("-filter_complex")
        .arg(filter)
        .arg("-loop")
        .arg("0")
        .arg(out_gif)
        .status()
        .with_context(|| format!("spawning ffmpeg for contact sheet ({})", ffmpeg.display()))?;
    if !status.success() {
        return Err(anyhow!("ffmpeg contact-sheet GIF failed ({status})"));
    }
    Ok(())
}
