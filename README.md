# LinguaCast

Dub any video into other languages **in the speaker's own voice**, on your laptop.

```
linguacast input.mp4 --langs es,zh,hi,fr,de,ja,pt,ar
```

Local-first by default. Single binary. Apache-2.0.

> **Status:** week-1 spike. Today the CLI supports `--langs es` end-to-end on a 60-second clip (Whisper-large-v3 → MADLAD-400 → Qwen3-TTS → ffmpeg mux). The other 7 languages are wired but will error this week. Lip-sync, consent gate, watermark, packaging, and more languages land in weeks 2–3 — see [OPE-6](https://example.invalid/OPE-6) for the milestone plan.

## Quickstart

```bash
# Build the CLI
cargo build --release

# One-time: install the Python sidecar (lazy-downloads models on first run)
cd sidecar
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd ..

# Dub a 60-second EN clip into Spanish
./target/release/linguacast samples/week1/input.mp4 --langs es \
  --i-understand-voice-clone-risks
```

Output lands in `linguacast-out/input.es.mp4`.

First run downloads ~6 GB of model weights to `~/.cache/linguacast/`. Subsequent runs reuse them and complete in roughly real-time on Apple Silicon.

## What's inside

| Stage | Model | License |
| --- | --- | --- |
| ASR (speech → text + timestamps) | [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | MIT |
| MT (EN → target) | [MADLAD-400-3B-MT](https://huggingface.co/google/madlad400-3b-mt) | Apache-2.0 |
| TTS (voice clone) | [Qwen3-TTS](https://huggingface.co/Qwen) | Apache-2.0 |
| Mux | ffmpeg | LGPL/GPL (system binary, not statically linked) |

Lip-sync is intentionally not included this week. It's [v0.2 scope](https://example.invalid/OPE-6).

## Why local-first

Cloud TTS APIs (ElevenLabs, OpenAI, Azure) work great but are paid, gate-kept, and slow to demo. LinguaCast's launch hook is *clone the repo, get a dubbed clip in three minutes, on the laptop you already own*. Every model in the pipeline has a fully-local path that works without an API key. Cloud is a `--cloud` flag, not a requirement — and it's not in scope until v0.2.

## Voice cloning safety

This is a real risk and we treat it as one.

- **Today (week 1):** the CLI refuses to produce dubbed audio without the `--i-understand-voice-clone-risks` flag. This is a placeholder gate. Do not ship audio without speaker consent.
- **Week 3 (launch-blocker):** the real consent gate ([OPE-12](https://example.invalid/OPE-12)) plus an inaudible perceptual watermark ([OPE-13](https://example.invalid/OPE-13)) become the default code path with no bypass.

If you find a way to remove the watermark or skip the consent gate, please open a security issue rather than publishing.

## License

Apache-2.0. Every dependency, transitively, is Apache-2.0 / MIT. The license audit lives in [LICENSES.md](LICENSES.md).
