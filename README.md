# LinguaCast

Dub any video into other languages **in the speaker's own voice**, on your laptop.

```
linguacast input.mp4 --langs es
```

Local-first. Single binary. Apache-2.0.

> **Status:** week-1 spike. EN→ES voice clone via
> [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) →
> [MADLAD-400-3B-MT](https://huggingface.co/google/madlad400-3b-mt) →
> [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) →
> ffmpeg mux. Other languages, lip-sync, consent gate, and packaging land
> in weeks 2–3.

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

### 3 — Pull model weights (once, ~10 GB)

This step downloads Whisper, MADLAD-400, and Qwen3-TTS into
`~/.cache/linguacast/`. It is a one-time step; the warm-cache dub below
does not re-download anything.

```bash
./target/release/linguacast pull
```

On a fast connection this takes 10–20 minutes. On an M1 the download is
I/O bound. Subsequent runs are warm-cache.

### 4 — Dub (warm cache — this is the headline TTW)

```bash
./target/release/linguacast samples/week1/input.mp4 --langs es \
  --i-understand-voice-clone-risks
```

Output lands in `linguacast-out/input.es.mp4`.

**Time-to-WOW from warm cache on M1:** roughly 1–2 minutes for a 60-second
clip (Whisper ~30s CPU int8, MADLAD ~TBD, Qwen3-TTS synthesis ~TBD).

---

## What's inside

| Stage | Model | License | Warm-cache latency (M1, 60s clip) |
| --- | --- | --- | --- |
| ASR (speech → text + timestamps) | [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) via faster-whisper | MIT | ~28s |
| MT (EN → target) | [MADLAD-400-3B-MT](https://huggingface.co/google/madlad400-3b-mt) | Apache-2.0 | pending |
| TTS (voice clone) | [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | Apache-2.0 | pending |
| Mux | ffmpeg | LGPL/GPL (system binary, not statically linked) | <1s |

Whisper runs CPU int8 on macOS (CTranslate2 has no Metal backend) at
~0.5× realtime. MPS/CUDA for MADLAD and Qwen3-TTS; CPU fallback if
unavailable.

Memory peaks per stage (sequential load/unload, M1):

| Stage | Peak RSS |
| --- | --- |
| Whisper large-v3 | ~3.8 GB |
| MADLAD-400-3B-MT | pending |
| Qwen3-TTS-1.7B-Base | pending |

Each model unloads before the next loads — only one model is resident at
a time. See `docs/engine-decision.md` for the full measurement log.

---

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

- **Today (week 1):** the CLI refuses to produce dubbed audio without the
  `--i-understand-voice-clone-risks` flag. This is a placeholder gate.
  Do not ship audio without speaker consent.
- **Week 3 (launch-blocker):** the real consent gate ([OPE-12]) plus an
  inaudible perceptual watermark ([OPE-13]) become the default code path
  with no bypass.

If you find a way to remove the watermark or skip the consent gate, please
open a security issue rather than publishing.

## License

Apache-2.0. Every dependency, transitively, is Apache-2.0, MIT, or
BSD-3-Clause. The Rust dep audit lives in [LICENSES.md](LICENSES.md) and
the model-weights audit lives in [MODELS.md](MODELS.md).
`scripts/check_model_licenses.py` enforces the model floor in CI
([OPE-17]).
