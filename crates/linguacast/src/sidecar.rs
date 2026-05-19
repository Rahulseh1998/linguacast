use crate::device::Device;
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use tracing::{debug, warn};

/// One JSON request per line on stdin, one JSON response per line on stdout.
/// Stderr is forwarded to the user's terminal — model download progress and
/// per-stage load times go there.
pub struct Sidecar {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Request<'a> {
    Hello,
    Pull {
        asr: &'a str,
        mt: &'a str,
        tts_size: &'a str,
    },
    Transcribe {
        audio_path: &'a Path,
        device: &'a str,
        asr: &'a str,
    },
    Translate {
        segments: &'a [Segment],
        source_lang: &'a str,
        target_lang: &'a str,
        mt: &'a str,
        device: &'a str,
    },
    Tts {
        segments: &'a [Segment],
        reference_audio_path: &'a Path,
        target_lang: &'a str,
        out_audio_path: &'a Path,
        target_duration_sec: f32,
        ref_text: &'a str,
        tts_size: &'a str,
        device: &'a str,
        engine: &'a str,
    },
    /// End-to-end dub: transcribe → translate → tts in one IPC call with
    /// sequential load/unload between each stage.
    RunDub {
        audio_path: &'a Path,
        reference_audio_path: &'a Path,
        target_lang: &'a str,
        out_audio_path: &'a Path,
        target_duration_sec: f32,
        asr: &'a str,
        mt: &'a str,
        tts_size: &'a str,
        device: &'a str,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub start: f32,
    pub end: f32,
    pub text: String,
}

#[derive(Debug, Deserialize)]
pub struct StageReport {
    pub name: String,
    pub model: String,
    pub stage_seconds: f64,
    pub peak_rss_mb: f64,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Response {
    Hello {
        version: String,
        torch_device: String,
        torch_version: String,
    },
    Pull {
        cache_root: String,
        models: BTreeMap<String, String>,
    },
    Transcribe {
        language: String,
        segments: Vec<Segment>,
        #[serde(default)]
        peak_rss_mb: f64,
    },
    Translate {
        segments: Vec<Segment>,
        #[serde(default)]
        peak_rss_mb: f64,
    },
    Tts {
        out_audio_path: PathBuf,
        duration_sec: f32,
        #[serde(default)]
        peak_rss_mb: f64,
    },
    RunDub {
        out_audio_path: PathBuf,
        duration_sec: f32,
        language: String,
        target_lang: String,
        segments: usize,
        segments_rendered: usize,
        stages: Vec<StageReport>,
        peak_rss_mb: f64,
    },
    Error {
        message: String,
        #[allow(dead_code)]
        recoverable: bool,
    },
}

impl Sidecar {
    /// Launch the Python sidecar. We try in order:
    ///   1. `--python` CLI arg if given
    ///   2. `$LINGUACAST_PYTHON`
    ///   3. `<repo>/sidecar/.venv/bin/python` if present
    ///   4. `python3` from PATH (last resort — likely missing deps)
    pub fn launch(python_override: Option<&Path>, script_dir: &Path) -> Result<Self> {
        let interpreter = resolve_python(python_override, script_dir)?;
        let script = script_dir.join("main.py");
        if !script.exists() {
            return Err(anyhow!(
                "sidecar entrypoint missing at {}",
                script.display()
            ));
        }
        debug!(
            "launching sidecar: {} {}",
            interpreter.display(),
            script.display()
        );
        let mut child = Command::new(&interpreter)
            .arg(&script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| {
                format!("spawning python sidecar at {}", interpreter.display())
            })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("sidecar missing stdin"))?;
        let stdout = BufReader::new(
            child
                .stdout
                .take()
                .ok_or_else(|| anyhow!("sidecar missing stdout"))?,
        );
        Ok(Self {
            child,
            stdin,
            stdout,
        })
    }

    pub fn send(&mut self, req: &Request) -> Result<Response> {
        let line = serde_json::to_string(req).context("encoding sidecar request")?;
        debug!("→ sidecar: {}", line);
        self.stdin
            .write_all(line.as_bytes())
            .and_then(|_| self.stdin.write_all(b"\n"))
            .and_then(|_| self.stdin.flush())
            .context("writing to sidecar stdin")?;

        let mut resp_line = String::new();
        let n = self
            .stdout
            .read_line(&mut resp_line)
            .context("reading sidecar stdout")?;
        if n == 0 {
            return Err(anyhow!(
                "sidecar closed stdout before responding (it may have crashed — check stderr above)"
            ));
        }
        debug!("← sidecar: {}", resp_line.trim_end());
        let resp: Response = serde_json::from_str(resp_line.trim_end())
            .with_context(|| format!("decoding sidecar response: {resp_line}"))?;
        Ok(resp)
    }

    pub fn hello(&mut self) -> Result<()> {
        match self.send(&Request::Hello)? {
            Response::Hello {
                version,
                torch_device,
                torch_version,
            } => {
                tracing::info!(
                    "sidecar ready · linguacast-sidecar {version} · torch {torch_version} · device {torch_device}"
                );
                Ok(())
            }
            Response::Error { message, .. } => Err(anyhow!("sidecar handshake failed: {message}")),
            other => Err(anyhow!("unexpected sidecar response to hello: {:?}", other)),
        }
    }

    pub fn pull(
        &mut self,
        asr: &str,
        mt: &str,
        tts_size: &str,
    ) -> Result<(String, BTreeMap<String, String>)> {
        match self.send(&Request::Pull {
            asr,
            mt,
            tts_size,
        })? {
            Response::Pull { cache_root, models } => Ok((cache_root, models)),
            Response::Error { message, .. } => Err(anyhow!("pull failed: {message}")),
            other => Err(anyhow!("unexpected pull response: {:?}", other)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_dub(
        &mut self,
        audio_path: &Path,
        reference_audio_path: &Path,
        target_lang: &str,
        out_audio_path: &Path,
        target_duration_sec: f32,
        asr: &str,
        mt: &str,
        tts_size: &str,
        device: &Device,
    ) -> Result<RunDubReport> {
        match self.send(&Request::RunDub {
            audio_path,
            reference_audio_path,
            target_lang,
            out_audio_path,
            target_duration_sec,
            asr,
            mt,
            tts_size,
            device: device.as_str(),
        })? {
            Response::RunDub {
                out_audio_path,
                duration_sec,
                language,
                target_lang,
                segments,
                segments_rendered,
                stages,
                peak_rss_mb,
            } => Ok(RunDubReport {
                out_audio_path,
                duration_sec,
                language,
                target_lang,
                segments,
                segments_rendered,
                stages,
                peak_rss_mb,
            }),
            Response::Error { message, .. } => Err(anyhow!("run_dub failed: {message}")),
            other => Err(anyhow!("unexpected run_dub response: {:?}", other)),
        }
    }
}

#[derive(Debug)]
pub struct RunDubReport {
    pub out_audio_path: PathBuf,
    pub duration_sec: f32,
    pub language: String,
    pub target_lang: String,
    pub segments: usize,
    pub segments_rendered: usize,
    pub stages: Vec<StageReport>,
    pub peak_rss_mb: f64,
}

impl Drop for Sidecar {
    fn drop(&mut self) {
        if let Err(e) = self.child.kill() {
            warn!("sidecar kill failed: {e}");
        }
        let _ = self.child.wait();
    }
}

fn resolve_python(python_override: Option<&Path>, script_dir: &Path) -> Result<PathBuf> {
    if let Some(p) = python_override {
        return Ok(p.to_path_buf());
    }
    if let Ok(p) = std::env::var("LINGUACAST_PYTHON") {
        return Ok(PathBuf::from(p));
    }
    let venv = script_dir.join(".venv/bin/python");
    if venv.exists() {
        return Ok(venv);
    }
    which::which("python3").map_err(|_| {
        anyhow!(
            "no Python interpreter found. Pass --python <path> or set LINGUACAST_PYTHON. \
             For a clean run: cd sidecar && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
        )
    })
}
