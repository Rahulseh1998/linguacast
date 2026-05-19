# LinguaCast Python sidecar

The Rust CLI orchestrates this directory as a subprocess. You do not interact
with it directly under normal use.

## Why a sidecar (and not a Rust-native rewrite)?

Whisper, MADLAD-400, and Qwen3-TTS have first-class Python implementations
on Hugging Face. Re-implementing them in Rust would add weeks of work and
zero user-visible value. The Voicebox release uses the same orchestrator +
sidecar pattern.

## What it does

One JSON request per line on stdin → one JSON response per line on stdout.
Stderr is reserved for human-readable progress (model downloads, timings).

Ops:
- `hello` — handshake, returns torch version and selected device.
- `transcribe` — Whisper-large-v3, returns segments with timestamps.
- `translate` — MADLAD-400-3B-MT, segment-by-segment.
- `tts` — Qwen3-TTS, voice-cloned synthesis in the target language. *(Week-1 work in progress — engine loader is wired; segment synth lands in the next commit.)*

## Install

```bash
cd sidecar
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The first end-to-end run downloads model weights to `~/.cache/linguacast/`
(or `$LINGUACAST_CACHE_DIR` if set). Roughly 6 GB for the full Whisper +
MADLAD + Qwen3-TTS stack. Subsequent runs reuse the cache.

## License

Apache-2.0 (matches the repo root). Every dependency in `requirements.txt`
is independently Apache-2.0 / MIT / BSD-3-Clause. See `../LICENSES.md` for
the audit.
