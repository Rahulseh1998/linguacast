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
        /// OPE-13 watermark id (32-bit, hex-encoded). When set, the
        /// sidecar embeds the perceptual watermark before writing the
        /// synthesized track to disk.
        #[serde(skip_serializing_if = "Option::is_none")]
        watermark_id: Option<&'a str>,
    },
    /// OPE-13 verifier: detect the perceptual watermark in a candidate
    /// audio file (no model load required, pure numpy+scipy).
    Verify {
        audio_path: &'a Path,
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
        #[serde(default)]
        segments_rendered: usize,
        #[serde(default)]
        watermark: serde_json::Value,
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
        #[serde(default)]
        watermark: serde_json::Value,
    },
    Verify {
        detected: bool,
        confidence: f64,
        #[serde(default)]
        watermark_id: Option<String>,
        #[serde(default)]
        version: Option<u64>,
        #[serde(default)]
        crc_ok: Option<bool>,
        #[serde(default)]
        repeats_voted: u64,
        #[serde(default)]
        bit_offset_frames: i64,
        sample_rate: u32,
        duration_sec: f64,
        elapsed_sec: f64,
        algorithm: String,
    },
    Progress {
        stage: String,
        phase: String,
        #[serde(default)]
        current: Option<u64>,
        #[serde(default)]
        total: Option<u64>,
        #[serde(default)]
        label: Option<String>,
    },
    Error {
        message: String,
        #[allow(dead_code)]
        recoverable: bool,
    },
}

/// Callback used by the orchestrator to surface in-flight progress.
/// Returning `()` keeps the API trivial; if we ever need cancellation we can
/// thread an `AtomicBool` through here instead.
pub type ProgressFn<'a> = &'a mut dyn FnMut(ProgressEvent);

#[derive(Debug, Clone)]
pub struct ProgressEvent {
    pub stage: String,
    pub phase: String,
    pub current: Option<u64>,
    pub total: Option<u64>,
    pub label: Option<String>,
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
        self.send_with_progress(req, &mut |_| {})
    }

    /// Send a request and drain `kind=progress` events to `on_progress` until
    /// a non-progress response is received.
    pub fn send_with_progress(
        &mut self,
        req: &Request,
        on_progress: ProgressFn<'_>,
    ) -> Result<Response> {
        let line = serde_json::to_string(req).context("encoding sidecar request")?;
        debug!("→ sidecar: {}", line);
        self.stdin
            .write_all(line.as_bytes())
            .and_then(|_| self.stdin.write_all(b"\n"))
            .and_then(|_| self.stdin.flush())
            .context("writing to sidecar stdin")?;

        loop {
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
            if let Response::Progress {
                stage,
                phase,
                current,
                total,
                label,
            } = resp
            {
                on_progress(ProgressEvent {
                    stage,
                    phase,
                    current,
                    total,
                    label,
                });
                continue;
            }
            return Ok(resp);
        }
    }

    pub fn transcribe(
        &mut self,
        audio_path: &Path,
        device: &Device,
        asr: &str,
        on_progress: ProgressFn<'_>,
    ) -> Result<TranscribeReport> {
        match self.send_with_progress(
            &Request::Transcribe {
                audio_path,
                device: device.as_str(),
                asr,
            },
            on_progress,
        )? {
            Response::Transcribe { language, segments, peak_rss_mb } => {
                Ok(TranscribeReport { language, segments, peak_rss_mb })
            }
            Response::Error { message, .. } => Err(anyhow!("transcribe failed: {message}")),
            other => Err(anyhow!("unexpected transcribe response: {:?}", other)),
        }
    }

    pub fn translate(
        &mut self,
        segments: &[Segment],
        source_lang: &str,
        target_lang: &str,
        mt: &str,
        device: &Device,
        on_progress: ProgressFn<'_>,
    ) -> Result<TranslateReport> {
        match self.send_with_progress(
            &Request::Translate {
                segments,
                source_lang,
                target_lang,
                mt,
                device: device.as_str(),
            },
            on_progress,
        )? {
            Response::Translate { segments, peak_rss_mb } => {
                Ok(TranslateReport { segments, peak_rss_mb })
            }
            Response::Error { message, .. } => Err(anyhow!("translate failed: {message}")),
            other => Err(anyhow!("unexpected translate response: {:?}", other)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn tts(
        &mut self,
        segments: &[Segment],
        reference_audio_path: &Path,
        target_lang: &str,
        out_audio_path: &Path,
        target_duration_sec: f32,
        ref_text: &str,
        tts_size: &str,
        device: &Device,
        watermark_id: Option<&str>,
        on_progress: ProgressFn<'_>,
    ) -> Result<TtsReport> {
        match self.send_with_progress(
            &Request::Tts {
                segments,
                reference_audio_path,
                target_lang,
                out_audio_path,
                target_duration_sec,
                ref_text,
                tts_size,
                device: device.as_str(),
                engine: "qwen3-tts",
            },
            on_progress,
        )? {
            Response::Tts {
                out_audio_path,
                duration_sec,
                peak_rss_mb,
                segments_rendered,
                watermark,
            } => Ok(TtsReport {
                out_audio_path,
                duration_sec,
                segments_rendered,
                peak_rss_mb,
                watermark,
            }),
            Response::Error { message, .. } => Err(anyhow!("tts failed: {message}")),
            other => Err(anyhow!("unexpected tts response: {:?}", other)),
        }
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

    /// OPE-13: detect the perceptual watermark in a candidate audio file.
    pub fn verify(&mut self, audio_path: &Path) -> Result<VerifyReport> {
        match self.send(&Request::Verify { audio_path })? {
            Response::Verify {
                detected,
                confidence,
                watermark_id,
                version,
                crc_ok,
                repeats_voted,
                bit_offset_frames,
                sample_rate,
                duration_sec,
                elapsed_sec,
                algorithm,
            } => Ok(VerifyReport {
                detected,
                confidence,
                watermark_id,
                version,
                crc_ok,
                repeats_voted,
                bit_offset_frames,
                sample_rate,
                duration_sec,
                elapsed_sec,
                algorithm,
            }),
            Response::Error { message, .. } => Err(anyhow!("verify failed: {message}")),
            other => Err(anyhow!("unexpected verify response: {:?}", other)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TranscribeReport {
    pub language: String,
    pub segments: Vec<Segment>,
    pub peak_rss_mb: f64,
}

#[derive(Debug, Clone)]
pub struct TranslateReport {
    pub segments: Vec<Segment>,
    pub peak_rss_mb: f64,
}

#[derive(Debug)]
pub struct TtsReport {
    pub out_audio_path: PathBuf,
    pub duration_sec: f32,
    pub segments_rendered: usize,
    pub peak_rss_mb: f64,
    pub watermark: serde_json::Value,
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
    pub watermark: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct VerifyReport {
    pub detected: bool,
    pub confidence: f64,
    pub watermark_id: Option<String>,
    pub version: Option<u64>,
    pub crc_ok: Option<bool>,
    pub repeats_voted: u64,
    pub bit_offset_frames: i64,
    pub sample_rate: u32,
    pub duration_sec: f64,
    pub elapsed_sec: f64,
    pub algorithm: String,
}

impl Drop for Sidecar {
    fn drop(&mut self) {
        if let Err(e) = self.child.kill() {
            warn!("sidecar kill failed: {e}");
        }
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Minimal mock sidecar that handles hello/transcribe/translate/tts and writes
    /// op counts to MOCK_OP_COUNT_FILE so the test can verify ASR runs exactly once.
    const MOCK_SIDECAR_PY: &str = r#"
import sys, json, os, wave, struct

count_file = os.environ.get("MOCK_OP_COUNT_FILE", "")
op_counts = {}

def write_counts():
    if count_file:
        with open(count_file, "w") as f:
            json.dump(op_counts, f)

def silent_wav(path, sr=24000, frames=240):
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack("<" + "h" * frames, *([0] * frames)))

for raw_line in sys.stdin:
    line = raw_line.strip()
    if not line:
        continue
    payload = json.loads(line)
    op = payload.get("op", "")
    op_counts[op] = op_counts.get(op, 0) + 1
    write_counts()

    if op == "hello":
        print(json.dumps({"kind": "hello", "version": "mock-1.0",
                          "torch_device": "cpu", "torch_version": "2.0"}), flush=True)
    elif op == "transcribe":
        print(json.dumps({
            "kind": "transcribe", "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
            "stage_seconds": 0.001, "peak_rss_mb": 10.0
        }), flush=True)
    elif op == "translate":
        tgt = payload.get("target_lang", "es")
        segs = payload.get("segments", [])
        out_segs = [{"start": s["start"], "end": s["end"], "text": f"({tgt}) {s['text']}"} for s in segs]
        print(json.dumps({
            "kind": "translate", "segments": out_segs,
            "stage_seconds": 0.001, "peak_rss_mb": 10.0
        }), flush=True)
    elif op == "tts":
        out = payload.get("out_audio_path", "/tmp/mock_tts_out.wav")
        silent_wav(out)
        print(json.dumps({
            "kind": "tts", "out_audio_path": out,
            "duration_sec": 0.01, "sample_rate": 24000,
            "segments_rendered": len(payload.get("segments", [])),
            "stage_seconds": 0.001, "peak_rss_mb": 10.0,
            "watermark": {"embedded": False}
        }), flush=True)
    else:
        print(json.dumps({"kind": "error", "message": f"unknown op: {op}",
                          "recoverable": True}), flush=True)
"#;

    fn find_python() -> PathBuf {
        for candidate in &["python3", "python"] {
            if let Ok(p) = which::which(candidate) {
                return p;
            }
        }
        PathBuf::from("python3")
    }

    /// Verify that the typed transcribe/translate/tts methods work correctly and
    /// that a two-language run calls ASR exactly once (the OPE-47 invariant).
    #[test]
    fn asr_once_two_langs() {
        let tmp = tempfile::TempDir::new().unwrap();

        let sidecar_script = tmp.path().join("main.py");
        std::fs::write(&sidecar_script, MOCK_SIDECAR_PY).unwrap();

        let count_file = tmp.path().join("op_counts.json");
        std::env::set_var("MOCK_OP_COUNT_FILE", count_file.display().to_string());

        // Minimal WAV for the audio path argument (mock doesn't read it).
        let audio = tmp.path().join("audio.wav");
        {
            use std::io::Write as _;
            let mut f = std::fs::File::create(&audio).unwrap();
            // 44-byte RIFF/WAV header with 0 data bytes
            f.write_all(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\
                         \x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\
                         \x02\x00\x10\x00data\x00\x00\x00\x00").unwrap();
        }

        let python = find_python();
        let mut sidecar = Sidecar::launch(Some(&python), tmp.path()).unwrap();
        sidecar.hello().unwrap();

        let device = Device::Cpu;

        // --- ASR once ---
        let asr = sidecar
            .transcribe(&audio, &device, "large-v3", &mut |_| {})
            .expect("transcribe should succeed");
        assert_eq!(asr.language, "en");
        assert!(!asr.segments.is_empty(), "mock should return at least one segment");

        // --- Two-lang MT+TTS loop reusing ASR segments ---
        for lang in &["es", "fr"] {
            let out = tmp.path().join(format!("out_{lang}.wav"));
            let tr = sidecar
                .translate(&asr.segments, "en", lang, "m2m100-418m", &device, &mut |_| {})
                .unwrap_or_else(|e| panic!("translate({lang}) failed: {e}"));
            assert!(!tr.segments.is_empty());

            let tts = sidecar
                .tts(
                    &tr.segments,
                    &audio,
                    lang,
                    &out,
                    1.0,
                    "hello world",
                    "1.7B",
                    &device,
                    None,
                    &mut |_| {},
                )
                .unwrap_or_else(|e| panic!("tts({lang}) failed: {e}"));
            assert_eq!(tts.segments_rendered, 1);
            assert!(out.exists(), "TTS should write the output WAV");
        }

        // Drop sidecar before reading count file (stdin close triggers flush).
        drop(sidecar);

        let counts_raw = std::fs::read_to_string(&count_file)
            .expect("mock sidecar should have written op count file");
        let counts: HashMap<String, u64> =
            serde_json::from_str(&counts_raw).expect("op count file should be valid JSON");

        assert_eq!(
            counts.get("transcribe").copied().unwrap_or(0),
            1,
            "ASR (transcribe) must run exactly once for multi-lang runs"
        );
        assert_eq!(
            counts.get("translate").copied().unwrap_or(0),
            2,
            "translate should run once per language (es + fr)"
        );
        assert_eq!(
            counts.get("tts").copied().unwrap_or(0),
            2,
            "tts should run once per language (es + fr)"
        );

        std::env::remove_var("MOCK_OP_COUNT_FILE");
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
