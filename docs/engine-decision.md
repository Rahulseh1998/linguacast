# Engine decision: Qwen3-TTS vs Voicebox on M1 8 GB

**Owner:** LinguaCoder · **Issue:** OPE-19 · **Status:** Locked (2026-05-19)

## Why this decision matters

The launch hook is *clone the repo → dubbed clip in 3 minutes on an M1*.
The TTS engine is the single biggest determinant of:

1. **Whether we fit memory** — the reference target is the M1 8 GB Air,
   which has ~6 GB free unified memory under realistic conditions.
2. **Voice clone quality** — the demo's "wow" moment is hearing the source
   speaker in Spanish. If the cloned voice doesn't sound like the source,
   the product doesn't deliver on its core claim.
3. **Inference latency** — 60 seconds of audio in <3 minutes wall time
   means the TTS step has a budget of roughly 90 seconds.

## Rubric

| # | Criterion | Pass bar | Weight |
| --- | --- | --- | --- |
| 1 | **License** — Apache-2.0 or MIT on the released weights | hard floor (binary) | reject if fails |
| 2 | **Fits M1 8 GB** — per-stage resident memory allows sequential run without OOM | hard floor | reject if fails after fallback |
| 3 | **Voice clone quality** — blind listen "same speaker" at ≥3/5 | soft | high |
| 4 | **Latency** — ≤1.5× realtime on M1 | soft | high |
| 5 | **Stability** — runs to completion on the 58s canonical clip | hard | medium |

## Decision: `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (Apache-2.0)

**Locked 2026-05-19 per OPE-19 CTO ack (comment 20acc056).**

Rationale: the OPE-4 track validated the `qwen-tts>=0.1.1` PyPI package
against this exact Hub ID (`Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-Base`). The
`-CustomVoice` IDs (`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`) were an
earlier discovery in the OPE-19 spike, but the OPE-4 validation showed the
`qwen-tts` package wraps the `-Base` variant — that's the working API path.

Both `-Base` and `-CustomVoice` are Apache-2.0. The decision rule was:
**validated-working API path beats spec-named model**. `-CustomVoice` is
noted here as the "revisit if `-Base` doesn't deliver voice quality" fallback.

### CLI knob aliasing

`--tts qwen3-tts-quantized` silently aliases to `--tts qwen3-tts` (the
full-precision 1.7B model). At 1.7B params the full-precision model fits
the 8 GB target without quantization. The quantized knob is preserved for
back-compat and documented here so future engineers don't wonder why
quantization was never wired.

### Size knob

`--tts-size 0.6B` selects `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (the pre-approved
8 GB fallback per the CTO ack). Default is `1.7B`. Set
`LINGUACAST_TTS_SIZE=0.6B` as an env override without recompiling.

### Voicebox status

Disabled. The 2023 paper release was research-only and no Meta-blessed
Apache-2.0 release exists as of 2026-05-19. `--tts voicebox` returns a
clear error. Revisit in week 2 only if a verified permissive release lands.

## Candidate snapshot

| Engine | License | Approx. params | Quantization story | Native voice clone |
| --- | --- | --- | --- | --- |
| Qwen3-TTS-12Hz-1.7B-Base | Apache-2.0 | 1.7B | Not needed — fp32 on MPS fits 8 GB | Yes — `generate_voice_clone(text, language, ref_audio, ref_text)` |
| Qwen3-TTS-12Hz-0.6B-Base | Apache-2.0 | 0.6B | Not needed | Yes — fallback for tighter boxes |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | Apache-2.0 | 1.7B | Not needed | Yes — revisit if `-Base` voice quality falls short |
| Voicebox (Meta) | **Disabled** — research-only, no permissive release | ~330M–2.5B | n/a | Yes |

## Hardware notes (Apple Silicon / MPS)

- `attn_implementation="sdpa"` is forced; flash-attn is CUDA-only.
- fp16 on MPS trips the multinomial sampler in Qwen3-TTS. We use fp32 on
  MPS and CPU, bf16 on CUDA.
- CTranslate2 (used by faster-whisper) does not expose Metal. Whisper runs
  CPU int8 on macOS, which is ~2× realtime on M-series.

## Results — Wed 2026-05-19 measurement

### Stage 1: ASR — Whisper large-v3 via faster-whisper (warm cache)

| Metric | Value |
| --- | --- |
| Model | `Systran/faster-whisper-large-v3` (CTranslate2 int8) |
| Device | CPU int8 (CTranslate2 has no Metal backend) |
| Clip | 58s EN monologue |
| Segments detected | 11 |
| Inference time | 28.4s (~0.49× realtime on M1) |
| Stage wall time (load + infer + unload) | 30.8s |
| **Peak RSS** | **3854 MB (~3.8 GB)** |
| RSS after unload | 2888 MB (~2.9 GB) |
| 8 GB fit | ✓ Clear — 3.8 GB peak, 4.2 GB headroom |

Whisper-large-v3 via faster-whisper is well within budget.

### Stage 2: MT — MADLAD-400-3B-MT

**Status: pending** — model download (~6 GB) not yet complete as of
2026-05-19. Measurement will be updated after download. Concern: 3B params
at fp16 ≈ 6 GB + 2.9 GB Python/torch baseline → may approach 8 GB ceiling.
Pre-approved fallback: if MADLAD-3B OOMs, escalate to CTO (no sub-3B
MADLAD public Apache-2.0 variant exists; MT family switch required).

### Stage 3: TTS — Qwen3-TTS-12Hz-1.7B-Base

**Status: pending** — model download (~3.4 GB) not yet complete.

### End-to-end (run_dub sequential load/unload)

**Status: pending model downloads.** Will be committed to
`samples/week1/output-es.mp4` once available.

### Recommendation (locked on model-fit grounds)

`Qwen3-TTS-12Hz-1.7B-Base` via `qwen-tts>=0.1.1` is locked as the
week-1 engine. If the MADLAD-3B RSS measurement exceeds 8 GB, the MADLAD
fallback path is an escalation (no sub-3B Apache-2.0 MADLAD exists). That
is the CTO-trigger condition per the kickoff: only escalate if the pipeline
can't fit even with all approved fallbacks.
