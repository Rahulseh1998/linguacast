# Week 1 sample

| File | Description |
|---|---|
| `input.mp4` | 58s EN monologue, single speaker, synthetic voice (macOS `say`) |
| `output-es.mp4` | EN→ES voice-cloned dub produced by LinguaCast |
| `regenerate.sh` | Reproduces `input.mp4` byte-for-byte (macOS + ffmpeg) |

## Regenerate `output-es.mp4`

From repo root (warm cache — run `linguacast pull` once first):

```bash
./target/release/linguacast samples/week1/input.mp4 --langs es \
  --i-understand-voice-clone-risks
```

Output lands at `linguacast-out/input.es.mp4`. Rename to `samples/week1/output-es.mp4` to match this fixture.

## Pipeline used to produce this sample

| Stage | Model | Time | Peak unified memory |
|---|---|---|---|
| ASR | `Systran/faster-whisper-large-v3` (CPU int8) | 29s | 3.86 GB |
| MT | `facebook/m2m100_418M` (MPS fp32) | 8s | 3.86 GB |
| TTS | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (MPS fp32) | 172s | 6.63 GB |

Total wall time: ~210s on M1. Peak memory stays under 7 GB at all times
(sequential load/unload — only one model resident at a time).
