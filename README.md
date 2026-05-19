# LinguaCast

Dub any video into other languages **in the speaker's own voice**, on your laptop.

```
linguacast input.mp4 --langs es,zh,hi,fr,de,ja,pt,ar,ko,ru,it,tr
```

Local-first. Single binary. Apache-2.0.

> **Status:** week-2. 12-language voice clone via
> [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) →
> [M2M-100-418M](https://huggingface.co/facebook/m2m100_418M) →
> [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) →
> ffmpeg mux. Lip-sync, consent gate, and Homebrew/pip packaging land
> in week 3.

---

## Quickstart

### 1 — Build

```bash
cargo build --release
```

### 2 — Install the Python sidecar (once)

```bash
cd sidecar
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd ..
```

### 3 — Dub (first run auto-pulls weights, ~10 GB)

```bash
./target/release/linguacast samples/week1/input.mp4 --langs es
```

The first invocation downloads Whisper, M2M-100, and Qwen3-TTS into
`~/.cache/linguacast/` automatically (10–20 minutes on a fast connection,
I/O bound on M1). Subsequent runs are warm-cache.

The first time you point LinguaCast at a new reference clip it prompts
for the voice-clone consent attestation — type `I AGREE` to confirm.
Consent is keyed on the audio bytes, so subsequent runs against the same
clip reuse it silently. For CI / batch pipelines pass
`--i-have-speaker-consent <path>` with a signed consent file. See
[`docs/consent-gate.md`](docs/consent-gate.md) for the full policy
(including the refusal list).

If you'd rather pre-download (useful for ops / CI), run the explicit
`linguacast pull` first. Output lands in `linguacast-out/input.<lang>.mp4`.

**Time-to-WOW from warm cache on M1:** ~3–4 minutes for a 60-second
clip (single-language). TTS synthesis is the bottleneck (~170s);
Whisper and translation together add ~40s. Fully local, no API key,
no cloud.

### 4 — Dub into all 12 launch languages + pack a shareable reel

```bash
./target/release/linguacast samples/week1/input.mp4 \
  --langs es,zh,hi,fr,de,ja,pt,ar,ko,ru,it,tr \
  --pack
```

`--pack` produces `linguacast-out/<stem>.pack.zip` containing all 12
MP4s, a 16:9 thumbnail per language, and a single contact-sheet GIF
cycling the outputs — ready to drop in a DM or social post.

---

## What's inside

| Stage | Model | License | Warm-cache latency (M1, 60s clip) |
| --- | --- | --- | --- |
| ASR (speech → text + timestamps) | [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) via faster-whisper | MIT | ~29s |
| MT (EN → target) | [M2M-100-418M](https://huggingface.co/facebook/m2m100_418M) (default, 12 langs) | MIT | ~8s |
| TTS (voice clone) | [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | Apache-2.0 | ~172s |
| Mux + pack | ffmpeg + zip | LGPL/GPL (system binary, not statically linked) | <2s |

Whisper runs CPU int8 on macOS (CTranslate2 has no Metal backend) at
~0.5× realtime. M2M-100 and Qwen3-TTS run on MPS (Apple Silicon GPU);
CPU fallback if unavailable.

Memory peaks per stage (sequential load/unload, M1 — measured 2026-05-19):

| Stage | Peak RSS | 8 GB M1 fit |
| --- | --- | --- |
| Whisper large-v3 | 3.86 GB | ✓ |
| M2M-100-418M | 3.86 GB | ✓ |
| Qwen3-TTS-1.7B-Base | 6.63 GB | ✓ (1.4 GB headroom) |

Pipeline ceiling: **6.63 GB**. Confirmed under `memory_pressure -l critical`.
Each model unloads before the next loads — only one model is resident at
a time. See `docs/engine-decision.md` for the full measurement log.

---

## Launch languages

The 12 languages wired for v0:

| Code | Language | M2M-100 | Qwen3-TTS native | Notes |
| --- | --- | --- | --- | --- |
| es | Spanish | ✓ | ✓ | Week-1 reference |
| zh | Chinese | ✓ | ✓ | |
| hi | Hindi | ✓ | ✓ (Auto) | Synthesis uses the TTS auto-lang path |
| fr | French | ✓ | ✓ | |
| de | German | ✓ | ✓ | |
| ja | Japanese | ✓ | ✓ | |
| pt | Portuguese | ✓ | ✓ | |
| ar | Arabic | ✓ | ✓ (Auto) | Synthesis uses the TTS auto-lang path |
| ko | Korean | ✓ | ✓ | |
| ru | Russian | ✓ | ✓ | |
| it | Italian | ✓ | ✓ | |
| tr | Turkish | ✓ | ✓ (Auto) | Synthesis uses the TTS auto-lang path |

## Fallback sizes

If you are on an 8 GB M1 and see OOM errors:

```bash
# Smaller Whisper (faster, slightly less accurate)
./target/release/linguacast input.mp4 --langs es --asr medium

# Smaller Qwen3-TTS (0.6B instead of 1.7B)
./target/release/linguacast input.mp4 --langs es --tts-size 0.6B
```

---

## Why local-first

Cloud TTS APIs (ElevenLabs, OpenAI, Azure) work great but are paid,
gate-kept, and slow to demo. LinguaCast's launch hook is *clone the repo,
get a dubbed clip in under three minutes, on the laptop you already own*.
Every model in the pipeline has a fully-local path that works without an
API key. Cloud is a future `--cloud` flag, not a requirement.

## Voice cloning safety

This is a real risk and we take it seriously.

LinguaCast will not produce voice-cloned output until you've attested
that you are the speaker, or you have written consent from the speaker.
The gate is on by default and has no bypass:

- **Interactive runs:** the CLI prints the attestation line on first use
  of a new reference clip and waits for you to type `I AGREE`. Consent
  is keyed on the SHA-256 of the reference audio and stored under
  `~/.linguacast/consents/<hash>.json`. Re-runs against the same audio
  reuse it silently; changing the audio bytes re-prompts.
- **CI / non-TTY runs:** pass `--i-have-speaker-consent <path>` with a
  signed consent file. The file must contain the attestation line
  verbatim; see [`docs/consent-gate.md`](docs/consent-gate.md) for the
  format.
- **Refusal list:** a small list of high-profile public figures whose
  voices have been most frequently abused in deepfake reports — runs
  matching the list are rejected regardless of consent. Update the list
  via PR against `crates/linguacast/data/refusal-list.json`.
- **Provenance — metadata:** every output MP4 ships with the consent
  hash, timestamp, signer, LinguaCast version, and watermark id in its
  container metadata (`comment` atom plus namespaced `linguacast_*`
  keys). Visible via `ffprobe -show_format`.
- **Provenance — audio watermark:** every voice-cloned audio track
  carries an inaudible perceptual watermark (patchwork
  spread-spectrum, 1 kHz–6 kHz band) encoding a 32-bit id derived
  deterministically from the consent hash. The watermark survives
  YouTube-style 1080p H.264 + AAC re-encode and `ffmpeg
  -map_metadata -1` metadata stripping at **100% detection / 80% id
  recovery** on the survival corpus
  ([`docs/watermark.md`](docs/watermark.md)). It's the *load-bearing*
  safety claim — metadata can be stripped; the audio watermark cannot.

```bash
# Check whether a file is a LinguaCast output and recover its provenance:
linguacast verify path/to/output.mp4
```

If you find a way to remove the watermark or skip the consent gate,
please open a security issue rather than publishing.

## License

Apache-2.0. Every dependency, transitively, is Apache-2.0, MIT, or
BSD-3-Clause. The Rust dep audit lives in [LICENSES.md](LICENSES.md) and
the model-weights audit lives in [MODELS.md](MODELS.md).
`scripts/check_model_licenses.py` enforces the model floor in CI
([OPE-17]).
