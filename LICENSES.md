# License audit

LinguaCast ships under **Apache-2.0**. Every dependency we depend on,
direct or transitive, is one of:

- Apache-2.0
- MIT (or MIT/X11)
- BSD-2-Clause / BSD-3-Clause
- ISC
- Unicode (data tables only)

Anything outside that floor is rejected by policy, including transitively.
See [OPE-6 milestones](https://example.invalid/OPE-6) for the rationale —
short version: this is an open project, the launch hook is "no friction,"
and a copyleft or non-commercial dep buried five levels deep is the kind
of footgun that ruins a launch.

## Models (lazy-downloaded, not bundled in the binary)

| Model | Hugging Face id | License | Used for |
| --- | --- | --- | --- |
| Whisper-large-v3 | `openai/whisper-large-v3` | MIT | ASR (speech → text + timestamps) |
| MADLAD-400-3B-MT | `google/madlad400-3b-mt` | Apache-2.0 | MT (source language → target) |
| Qwen3-TTS | `Qwen/Qwen3-TTS` *(and `Qwen3-TTS-Quantized`)* | Apache-2.0 | Voice-clone TTS |

**Explicitly rejected:**

- **NLLB (`facebook/nllb-200-*`)** — CC-BY-NC-4.0. Non-commercial means
  not us. MADLAD-400 covers the same languages at comparable quality and
  is Apache-2.0.
- **Voicebox** *(Meta, 2023 paper + 2024 release)* — license still under
  review. Wired into the CLI as an `--tts voicebox` opt-in but the
  sidecar refuses to load it until we can confirm Apache-2.0 / MIT (or
  equivalent) on the released weights. Decision lives on [OPE-19](https://example.invalid/OPE-19).
- **fairseq** — code is MIT, but the default model card points at NLLB,
  so depending on it creates a license footgun. We import the specific
  transformers loaders we need directly instead.

## Rust crates (workspace deps)

Audited via `cargo metadata` against the workspace's direct deps. Run
`cargo deny check licenses` from a future CI job to enforce the floor.

| Crate | Version | License | Why we need it |
| --- | --- | --- | --- |
| anyhow | 1.x | MIT OR Apache-2.0 | error plumbing |
| thiserror | 2.x | MIT OR Apache-2.0 | derive-based errors |
| clap | 4.x | MIT OR Apache-2.0 | CLI parsing |
| serde / serde_json | 1.x | MIT OR Apache-2.0 | JSON IPC with sidecar |
| tracing / tracing-subscriber | 0.1.x / 0.3.x | MIT | structured logging |
| which | 8.x | MIT | locate ffmpeg / python on PATH |
| tempfile | 3.x | MIT OR Apache-2.0 | scratch dirs for pipeline |
| directories | 6.x | MIT OR Apache-2.0 | platform-aware cache paths |
| sha2 | 0.10.x | MIT OR Apache-2.0 | SHA-256 of reference audio + consent file (OPE-12 consent gate) |
| indicatif | 0.17.x | MIT | per-language progress bars (OPE-44 UX) |
| zip | 2.x | MIT | `--pack` zip writer (no default features; deflate only) |

All of the above are MIT or MIT/Apache dual-licensed. Transitive deps are
audited as part of the cargo-deny step (TODO: wire `deny.toml` into CI in
week 2 — out of scope for the week-1 spike, but the floor is enforced by
the explicit dep list).

## Python sidecar (pip deps)

See [`sidecar/requirements.txt`](sidecar/requirements.txt) for the
authoritative list. Versions are pinned to permissive-licensed releases.

| Package | License | Why |
| --- | --- | --- |
| torch / torchaudio | BSD-3-Clause / BSD-2-Clause | tensors + audio I/O |
| transformers | Apache-2.0 | MADLAD / Qwen3-TTS loaders (Whisper now via `openai-whisper`) |
| openai-whisper | MIT | ASR (Whisper-large-v3 long-form) |
| tiktoken | MIT | tokenizer pulled by openai-whisper |
| accelerate | Apache-2.0 | device dispatch + memory-efficient init |
| sentencepiece | Apache-2.0 | tokenizer for T5/MADLAD |
| librosa | ISC | audio resampling |
| numpy | BSD-3-Clause | linear algebra |
| soundfile | BSD-3-Clause | WAV read/write |
| huggingface-hub | Apache-2.0 | model download client |
| tqdm | MIT or MPL-2.0 (dual) | first-run download progress |

**Not used (rejected or avoided):**

- `fairseq` — see note above. Pulls NLLB.
- `bitsandbytes` — MIT, but ships GPU-only kernels that complicate the CPU
  fallback story. If we need int8 quantization we'll prefer `transformers`'
  built-in `bitsandbytes` integration *with explicit guardrails*, or move
  to GGUF.

## System binaries (not bundled)

- **ffmpeg** — LGPL by default, GPL with `--enable-gpl` codecs. We invoke
  the user's system binary; we do not statically link. The Homebrew /
  apt builds are LGPL-compatible. This is the standard pattern: ffmpeg
  is a runtime dep declared in `brew install` and `apt install` instructions,
  not bundled.

## Audit policy

When you add a new dependency:

1. Check its license in its repo / on crates.io / on PyPI.
2. If it's MIT / Apache-2.0 / BSD / ISC, add it here with a one-line
   justification.
3. If it's anything else, **do not add it**. Open an issue and ask CTO.
4. If it pulls a non-permissive transitive (e.g. fairseq → NLLB), it's
   blocked. Find an alternative.
