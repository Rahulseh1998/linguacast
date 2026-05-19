//! Voice-clone consent gate. Launch-blocker per OPE-12.
//!
//! Before any TTS render, the speaker (or an authorised consenter) must
//! attest in writing that they consent to the voice clone. The attestation
//! is keyed on the SHA-256 of the reference audio so renaming the file
//! does not bypass the gate; it is re-prompted whenever the audio bytes
//! change. A small refusal list rejects voice-cloning of high-profile
//! public figures regardless of attestation.
//!
//! Records live under `~/.linguacast/consents/<sha256>.json`. Every output
//! MP4 carries the consent hash + timestamp in its container metadata so
//! downstream tooling can verify provenance.

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{self, BufRead, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// One-line attestation the speaker (or consenter) must agree to.
pub const ATTESTATION_PHRASE: &str =
    "I am the speaker in this audio, or I have written consent from the speaker.";

/// Word the user types in interactive mode to confirm. Stored in the
/// consent record so a reviewer can tell interactive from non-interactive
/// flows apart.
pub const AGREE_TOKEN: &str = "I AGREE";

/// Embedded starter refusal list. Override with `--refusal-list <path>`.
const REFUSAL_LIST_BUILTIN: &str = include_str!("../data/refusal-list.json");

const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConsentMode {
    Interactive,
    NonInteractiveFile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRecord {
    pub schema: u32,
    pub audio_sha256: String,
    pub attestation: String,
    pub agree_token: String,
    pub speaker_name: Option<String>,
    pub signed_by: String,
    pub hostname: String,
    pub machine_fingerprint: String,
    pub timestamp_unix: u64,
    pub timestamp_iso: String,
    pub linguacast_version: String,
    pub mode: ConsentMode,
    pub consent_file_path: Option<PathBuf>,
    pub consent_file_sha256: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RefusalList {
    #[allow(dead_code)]
    pub version: u32,
    #[allow(dead_code)]
    pub updated_at: String,
    pub model_card_url: String,
    #[allow(dead_code)]
    pub policy: String,
    pub entries: Vec<RefusalEntry>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RefusalEntry {
    pub name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub audio_sha256: Vec<String>,
    #[serde(default)]
    pub reason: String,
}

/// Options driving `obtain_consent`. Kept as a struct so tests can inject
/// non-interactive stdin without touching the real terminal.
pub struct ConsentOpts<'a> {
    pub reference_audio: &'a Path,
    pub speaker_name: Option<&'a str>,
    pub consent_file: Option<&'a Path>,
    pub consent_store_dir: PathBuf,
    pub refusal_list_override: Option<&'a Path>,
    pub force_non_interactive: bool,
}

/// Outcome of the consent flow — distinguishes "first-time signed this
/// heartbeat" from "loaded from the store" so the CLI can tell the user
/// which happened.
#[derive(Debug)]
pub enum ConsentOutcome {
    Reused(ConsentRecord),
    NewlySigned(ConsentRecord),
}

impl ConsentOutcome {
    #[allow(dead_code)]
    pub fn record(&self) -> &ConsentRecord {
        match self {
            ConsentOutcome::Reused(r) => r,
            ConsentOutcome::NewlySigned(r) => r,
        }
    }
}

/// Compute the SHA-256 of a file streamed in 64 KiB chunks. We deliberately
/// don't load the file fully; reference clips can be tens of MB.
pub fn compute_audio_hash(path: &Path) -> Result<String> {
    let mut file = File::open(path)
        .with_context(|| format!("opening reference audio {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("reading reference audio {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_encode(&hasher.finalize()))
}

/// SHA-256 of arbitrary bytes (used to hash the consent file body too).
pub fn sha256_bytes(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex_encode(&h.finalize())
}

pub fn load_refusal_list(override_path: Option<&Path>) -> Result<RefusalList> {
    if let Some(p) = override_path {
        let raw = fs::read_to_string(p)
            .with_context(|| format!("reading refusal list {}", p.display()))?;
        return serde_json::from_str(&raw)
            .with_context(|| format!("parsing refusal list {}", p.display()));
    }
    serde_json::from_str(REFUSAL_LIST_BUILTIN)
        .context("parsing embedded refusal list (this is a build-time invariant)")
}

#[derive(Debug)]
pub struct RefusalHit<'a> {
    pub entry: &'a RefusalEntry,
    pub matched_on: &'static str,
    pub model_card_url: &'a str,
}

/// Check the refusal list against an optional self-declared speaker name
/// AND the reference-audio hash. Returns `Ok(None)` for clean runs.
pub fn check_refusal_list<'a>(
    list: &'a RefusalList,
    speaker_name: Option<&str>,
    audio_hash: &str,
) -> Option<RefusalHit<'a>> {
    let name_norm = speaker_name.map(normalize_name);
    for e in &list.entries {
        if let Some(n) = &name_norm {
            if normalize_name(&e.name) == *n {
                return Some(RefusalHit {
                    entry: e,
                    matched_on: "name",
                    model_card_url: &list.model_card_url,
                });
            }
            for alias in &e.aliases {
                if normalize_name(alias) == *n {
                    return Some(RefusalHit {
                        entry: e,
                        matched_on: "alias",
                        model_card_url: &list.model_card_url,
                    });
                }
            }
        }
        if e.audio_sha256.iter().any(|h| h.eq_ignore_ascii_case(audio_hash)) {
            return Some(RefusalHit {
                entry: e,
                matched_on: "audio_sha256",
                model_card_url: &list.model_card_url,
            });
        }
    }
    None
}

fn normalize_name(s: &str) -> String {
    s.trim().to_lowercase()
}

/// Default consent store: `~/.linguacast/consents/`. Honours the
/// `LINGUACAST_HOME` env var so tests and CI can scope it.
pub fn default_consent_dir() -> Result<PathBuf> {
    if let Ok(custom) = std::env::var("LINGUACAST_HOME") {
        return Ok(PathBuf::from(custom).join("consents"));
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(|| directories::BaseDirs::new().map(|b| b.home_dir().to_path_buf()))
        .ok_or_else(|| anyhow!("could not determine home directory; set LINGUACAST_HOME"))?;
    Ok(home.join(".linguacast").join("consents"))
}

fn consent_path(store_dir: &Path, hash: &str) -> PathBuf {
    store_dir.join(format!("{hash}.json"))
}

fn load_existing(store_dir: &Path, hash: &str) -> Result<Option<ConsentRecord>> {
    let p = consent_path(store_dir, hash);
    if !p.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(&p)
        .with_context(|| format!("reading consent record {}", p.display()))?;
    let rec: ConsentRecord = serde_json::from_str(&raw)
        .with_context(|| format!("parsing consent record {}", p.display()))?;
    if rec.audio_sha256 != hash {
        bail!(
            "consent record at {} has hash {} but file is keyed on {}",
            p.display(),
            rec.audio_sha256,
            hash
        );
    }
    Ok(Some(rec))
}

fn save_record(store_dir: &Path, rec: &ConsentRecord) -> Result<PathBuf> {
    fs::create_dir_all(store_dir)
        .with_context(|| format!("creating consent dir {}", store_dir.display()))?;
    let p = consent_path(store_dir, &rec.audio_sha256);
    let json = serde_json::to_string_pretty(rec).context("serialising consent record")?;
    fs::write(&p, json).with_context(|| format!("writing consent record {}", p.display()))?;
    Ok(p)
}

/// Orchestrate the consent flow end-to-end. This is the only entrypoint
/// the rest of the pipeline should call.
pub fn obtain_consent(opts: &ConsentOpts) -> Result<ConsentOutcome> {
    let hash = compute_audio_hash(opts.reference_audio)?;
    let list = load_refusal_list(opts.refusal_list_override)?;
    if let Some(hit) = check_refusal_list(&list, opts.speaker_name, &hash) {
        bail!(
            "refusal-list match: {name} (matched on {matched}). Reason: {reason}\n\n\
             LinguaCast refuses voice-cloning of this reference regardless of consent. \
             See the model card: {model_card}",
            name = hit.entry.name,
            matched = hit.matched_on,
            reason = hit.entry.reason,
            model_card = hit.model_card_url,
        );
    }

    if let Some(existing) = load_existing(&opts.consent_store_dir, &hash)? {
        return Ok(ConsentOutcome::Reused(existing));
    }

    let record = if let Some(consent_file) = opts.consent_file {
        sign_from_file(&hash, opts.speaker_name, consent_file)?
    } else if opts.force_non_interactive || !io::stdin().is_terminal() {
        bail!(
            "voice-clone consent required (no record for this reference audio).\n\
             stdin is not a TTY, so the interactive attestation cannot run.\n\
             Re-run with --i-have-speaker-consent <path-to-consent-file>.\n\
             The consent file must contain the line:\n  {phrase}\n",
            phrase = ATTESTATION_PHRASE
        );
    } else {
        sign_interactively(&hash, opts.speaker_name)?
    };

    save_record(&opts.consent_store_dir, &record)?;
    Ok(ConsentOutcome::NewlySigned(record))
}

fn sign_from_file(
    hash: &str,
    speaker_name: Option<&str>,
    consent_file: &Path,
) -> Result<ConsentRecord> {
    let bytes = fs::read(consent_file)
        .with_context(|| format!("reading consent file {}", consent_file.display()))?;
    let body = String::from_utf8_lossy(&bytes);
    if !body
        .lines()
        .any(|l| l.trim().eq_ignore_ascii_case(ATTESTATION_PHRASE))
    {
        bail!(
            "consent file {} does not contain the required attestation line.\n\
             It must include, verbatim on its own line:\n  {phrase}",
            consent_file.display(),
            phrase = ATTESTATION_PHRASE
        );
    }
    let signer_from_file = body
        .lines()
        .find_map(|l| {
            let t = l.trim();
            t.strip_prefix("signer:")
                .or_else(|| t.strip_prefix("Signer:"))
                .map(|s| s.trim().to_string())
        });
    let file_hash = sha256_bytes(&bytes);
    Ok(new_record(
        hash,
        speaker_name.map(str::to_string).or(signer_from_file),
        ConsentMode::NonInteractiveFile,
        Some(consent_file.to_path_buf()),
        Some(file_hash),
    ))
}

fn sign_interactively(hash: &str, speaker_name: Option<&str>) -> Result<ConsentRecord> {
    let mut stderr = io::stderr().lock();
    let stdin = io::stdin();
    writeln!(stderr)?;
    writeln!(stderr, "── Voice-clone consent required ──")?;
    writeln!(stderr)?;
    writeln!(
        stderr,
        "  Reference audio SHA-256: {hash}\n"
    )?;
    writeln!(
        stderr,
        "Before LinguaCast produces voice-cloned output you must attest:"
    )?;
    writeln!(stderr, "\n    \"{ATTESTATION_PHRASE}\"\n")?;
    writeln!(
        stderr,
        "Type {AGREE_TOKEN} to confirm, then press Enter (Ctrl-C to abort):"
    )?;
    write!(stderr, "> ")?;
    stderr.flush()?;

    let mut answer = String::new();
    let n = stdin.lock().read_line(&mut answer).context("reading attestation from stdin")?;
    if n == 0 {
        bail!("stdin closed before attestation — aborting");
    }
    let answer_norm = answer.trim();
    if !answer_norm.eq_ignore_ascii_case(AGREE_TOKEN) {
        bail!(
            "consent not granted (you typed {answer:?}, expected {AGREE_TOKEN}). \
             No voice-clone output produced.",
            answer = answer_norm
        );
    }

    let signer = match speaker_name {
        Some(n) => Some(n.to_string()),
        None => {
            write!(stderr, "Your name (optional, for the audit log): ")?;
            stderr.flush()?;
            let mut name = String::new();
            let _ = stdin.lock().read_line(&mut name);
            let name = name.trim();
            if name.is_empty() {
                None
            } else {
                Some(name.to_string())
            }
        }
    };

    writeln!(
        stderr,
        "\nConsent recorded. Hash stored in your local consent store; re-runs against the same audio will reuse it.\n"
    )?;

    Ok(new_record(
        hash,
        signer,
        ConsentMode::Interactive,
        None,
        None,
    ))
}

fn new_record(
    hash: &str,
    speaker_name: Option<String>,
    mode: ConsentMode,
    consent_file_path: Option<PathBuf>,
    consent_file_sha256: Option<String>,
) -> ConsentRecord {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let user = std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "unknown".to_string());
    let host = hostname();
    let fp = sha256_bytes(format!("{user}@{host}/{}", std::env::consts::OS).as_bytes());
    let signed_by = format!("{user}@{host}");

    ConsentRecord {
        schema: SCHEMA_VERSION,
        audio_sha256: hash.to_string(),
        attestation: ATTESTATION_PHRASE.to_string(),
        agree_token: AGREE_TOKEN.to_string(),
        speaker_name,
        signed_by,
        hostname: host,
        machine_fingerprint: fp,
        timestamp_unix: now,
        timestamp_iso: iso_utc(now),
        linguacast_version: env!("CARGO_PKG_VERSION").to_string(),
        mode,
        consent_file_path,
        consent_file_sha256,
    }
}

fn hostname() -> String {
    if let Ok(h) = std::env::var("HOSTNAME") {
        if !h.is_empty() {
            return h;
        }
    }
    if let Ok(out) = std::process::Command::new("hostname").output() {
        if out.status.success() {
            let h = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !h.is_empty() {
                return h;
            }
        }
    }
    if let Ok(s) = std::fs::read_to_string("/etc/hostname") {
        let s = s.trim();
        if !s.is_empty() {
            return s.to_string();
        }
    }
    "unknown".to_string()
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// UNIX seconds → "YYYY-MM-DDTHH:MM:SSZ" without pulling in chrono.
fn iso_utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let mut tod = secs % 86_400;
    let h = tod / 3600;
    tod %= 3600;
    let m = tod / 60;
    let s = tod % 60;
    let (y, mo, d) = civil_from_days(days);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

/// Howard Hinnant, "chrono-Compatible Low-Level Date Algorithms".
fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as i64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m as u32, d as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;
    use tempfile::tempdir;

    fn write_tmp(dir: &Path, name: &str, contents: &[u8]) -> PathBuf {
        let p = dir.join(name);
        let mut f = File::create(&p).unwrap();
        f.write_all(contents).unwrap();
        p
    }

    #[test]
    fn hash_is_stable_and_streaming() {
        let d = tempdir().unwrap();
        let p = write_tmp(d.path(), "ref.wav", b"hello world");
        let h = compute_audio_hash(&p).unwrap();
        assert_eq!(
            h,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn embedded_refusal_list_parses() {
        let list = load_refusal_list(None).unwrap();
        assert!(!list.entries.is_empty());
        assert!(list.model_card_url.starts_with("http"));
    }

    #[test]
    fn refusal_list_hits_on_name_alias_and_hash() {
        let raw = serde_json::json!({
            "version": 1,
            "updated_at": "2026-01-01",
            "model_card_url": "https://example.invalid/card",
            "policy": "test",
            "entries": [
                {
                    "name": "Test Person",
                    "aliases": ["the test person", "T. Person"],
                    "audio_sha256": ["abc123"],
                    "reason": "unit test"
                }
            ]
        });
        let list: RefusalList = serde_json::from_value(raw).unwrap();
        assert!(check_refusal_list(&list, Some("test person"), "deadbeef").is_some());
        assert!(check_refusal_list(&list, Some("THE Test Person"), "deadbeef").is_some());
        assert!(check_refusal_list(&list, None, "ABC123").is_some());
        assert!(check_refusal_list(&list, Some("Random Other"), "deadbeef").is_none());
    }

    #[test]
    fn consent_file_must_contain_phrase() {
        let d = tempdir().unwrap();
        let ref_audio = write_tmp(d.path(), "ref.wav", b"hi");
        let store = d.path().join("store");
        let bad = write_tmp(d.path(), "consent.txt", b"I solemnly swear nothing.");
        let res = obtain_consent(&ConsentOpts {
            reference_audio: &ref_audio,
            speaker_name: None,
            consent_file: Some(&bad),
            consent_store_dir: store,
            refusal_list_override: None,
            force_non_interactive: true,
        });
        let err = res.unwrap_err().to_string();
        assert!(
            err.contains("does not contain the required attestation line"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn non_tty_without_consent_file_errors_helpfully() {
        let d = tempdir().unwrap();
        let ref_audio = write_tmp(d.path(), "ref.wav", b"hi");
        let res = obtain_consent(&ConsentOpts {
            reference_audio: &ref_audio,
            speaker_name: None,
            consent_file: None,
            consent_store_dir: d.path().join("store"),
            refusal_list_override: None,
            force_non_interactive: true,
        });
        let err = res.unwrap_err().to_string();
        assert!(err.contains("--i-have-speaker-consent"), "got: {err}");
        assert!(err.contains(ATTESTATION_PHRASE), "got: {err}");
    }

    #[test]
    fn consent_is_cached_by_audio_hash() {
        let d = tempdir().unwrap();
        let ref_audio = write_tmp(d.path(), "ref.wav", b"hi");
        let store = d.path().join("store");
        let good = write_tmp(
            d.path(),
            "consent.txt",
            format!("{ATTESTATION_PHRASE}\nsigner: Alice\n").as_bytes(),
        );

        let first = obtain_consent(&ConsentOpts {
            reference_audio: &ref_audio,
            speaker_name: None,
            consent_file: Some(&good),
            consent_store_dir: store.clone(),
            refusal_list_override: None,
            force_non_interactive: true,
        })
        .unwrap();
        assert!(matches!(first, ConsentOutcome::NewlySigned(_)));
        let rec = first.record();
        assert_eq!(rec.speaker_name.as_deref(), Some("Alice"));
        assert_eq!(rec.mode, ConsentMode::NonInteractiveFile);
        assert!(rec.consent_file_sha256.is_some());

        // Same audio → reuse from store, no consent-file needed second time.
        let second = obtain_consent(&ConsentOpts {
            reference_audio: &ref_audio,
            speaker_name: None,
            consent_file: None,
            consent_store_dir: store,
            refusal_list_override: None,
            force_non_interactive: true,
        })
        .unwrap();
        assert!(matches!(second, ConsentOutcome::Reused(_)));
        assert_eq!(second.record().audio_sha256, rec.audio_sha256);
    }

    #[test]
    fn changed_reference_audio_reprompts() {
        let d = tempdir().unwrap();
        let ref_a = write_tmp(d.path(), "a.wav", b"AAA");
        let ref_b = write_tmp(d.path(), "b.wav", b"BBB");
        let store = d.path().join("store");
        let consent = write_tmp(
            d.path(),
            "consent.txt",
            format!("{ATTESTATION_PHRASE}\n").as_bytes(),
        );

        obtain_consent(&ConsentOpts {
            reference_audio: &ref_a,
            speaker_name: None,
            consent_file: Some(&consent),
            consent_store_dir: store.clone(),
            refusal_list_override: None,
            force_non_interactive: true,
        })
        .unwrap();

        // Different audio → no cached consent → must re-prompt (non-tty errors).
        let res = obtain_consent(&ConsentOpts {
            reference_audio: &ref_b,
            speaker_name: None,
            consent_file: None,
            consent_store_dir: store,
            refusal_list_override: None,
            force_non_interactive: true,
        });
        assert!(res.is_err(), "second audio should not reuse first's consent");
    }

    #[test]
    fn celebrity_name_is_refused_via_override_list() {
        let d = tempdir().unwrap();
        let ref_audio = write_tmp(d.path(), "ref.wav", b"hi");
        let list = write_tmp(
            d.path(),
            "list.json",
            br#"{
              "version": 1,
              "updated_at": "2026-01-01",
              "model_card_url": "https://example.invalid/card",
              "policy": "test",
              "entries": [{"name": "Famous Person", "aliases": [], "audio_sha256": [], "reason": "unit test"}]
            }"#,
        );
        let res = obtain_consent(&ConsentOpts {
            reference_audio: &ref_audio,
            speaker_name: Some("Famous Person"),
            consent_file: None,
            consent_store_dir: d.path().join("store"),
            refusal_list_override: Some(&list),
            force_non_interactive: true,
        });
        let err = res.unwrap_err().to_string();
        assert!(err.contains("refusal-list match"), "got: {err}");
        assert!(err.contains("Famous Person"), "got: {err}");
        assert!(err.contains("https://example.invalid/card"), "got: {err}");
    }

    #[test]
    fn iso_utc_known_values() {
        assert_eq!(iso_utc(0), "1970-01-01T00:00:00Z");
        // 2000-01-01T00:00:00Z — the canonical reference moment used in many
        // RFC tests; off-by-one bugs in civil_from_days surface here.
        assert_eq!(iso_utc(946_684_800), "2000-01-01T00:00:00Z");
        // 2024-02-29T12:34:56Z — exercises Feb 29 on a leap year.
        assert_eq!(iso_utc(1_709_210_096), "2024-02-29T12:34:56Z");
    }
}
