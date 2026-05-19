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
| Inference time | 27–29s (~0.47–0.50× realtime on M1, 2 runs) |
| Stage wall time (load + infer + unload) | 29–31s |
| **Peak RSS** | **3854–3861 MB (~3.86 GB)** |
| RSS after unload | 2888 MB (~2.9 GB) |
| 8 GB fit | ✓ Clear — 3.8 GB peak, 4.2 GB headroom |

Whisper-large-v3 via faster-whisper is well within budget.

### Memory measurement methodology

**The trap:** `resource.getrusage(RUSAGE_SELF).ru_maxrss` (and `psutil`'s
process RSS) report only CPU-mapped pages. On Apple Silicon, MPS tensors
live in unified memory but are GPU-mapped; they don't show up in either
counter. A model that has eaten 11 GB of unified memory can still report
"500 MB RSS" — which is how this track nearly shipped a sub-8 GB claim
that would have OOMed the moment a 1.7B model loaded on the demo machine.

**The fix:** read the IOAccelerator alloc counter directly:

```bash
ioreg -r -c IOAccelerator | grep '"Alloc system memory"'
```

That value is the source-of-truth for unified-memory allocation on macOS
and is what the `peak_rss_mb` numbers in the pipeline JSON now report on
Apple Silicon. CPU-only systems still report `ru_maxrss`; the gap matters
only for MPS-resident models.

### Stage 2: MT — engine choice

**Selected: `facebook/m2m100_418M` (MIT)** — locked per CTO ack 2026-05-19
(comment [73a125be](#)).

MADLAD-400-3B-MT was the original choice but fails the 8 GB M1 ceiling
on both supported dtypes:

| Config | System baseline | MADLAD MPS delta | Peak total | 8 GB fits? |
|---|---|---|---|---|
| MADLAD-3B fp16 (MPS) | ~3.2 GB | +8.2 GB | **11.4 GB** | ✗ |
| MADLAD-3B fp32 (MPS) | ~3.2 GB | +11.4 GB | **14.6 GB** | ✗ |

- fp16 on MPS trips T5's multinomial sampler (repetition loops); fp32
  is required for translation quality. Both exceed 8 GB on the demo box.
- The OPE-19 kickoff pre-approved `madlad-1B` as a fallback, but **no
  public Apache-2.0 sub-3B MADLAD release exists** (confirmed against the
  Google MADLAD repo, the HF org page, and the paper's release list).
  The MADLAD family floor on a permissive license is the 3B variant.
- This triggered the launch-hook escalation. CTO approved the MT family
  switch to M2M-100 in comment 73a125be.

**Sibling-license note:** `facebook/m2m100_418M` is **MIT**. Its sibling
model `facebook/nllb-200-*` is **CC-BY-NC** and is rejected by
`scripts/check_model_licenses.py`. Don't confuse the two — they are
similar-shaped Facebook multilingual MT models and easy to swap by
accident. M2M-100 = ship; NLLB = block.

**Opt-in MADLAD-3B:** still exposed as `--mt madlad-3b` for users on
≥16 GB hosts. The runtime forces it to CPU with bf16 (≈6 GB resident +
activations), since the MPS path OOMs.

**Optional upgrades the CTO authorized (half-day budget):**
- `MADLAD-3B-CPU-bf16` end-to-end under `memory_pressure -l critical` —
  if it actually fits and quality wins, default could swap.
- `Helsinki-NLP/opus-mt-en-es` MarianMT — ~300 MB, EN→ES specialist.
  Verify per-model-card license (Apache-2.0 on this one) before A/B.

### Stage 3: TTS — Qwen3-TTS-12Hz-1.7B-Base

Locked. Full repo download (13 files including `speech_tokenizer/
preprocessor_config.json`) is enforced via `huggingface_hub.snapshot_download`
at the top of `_load_qwen_tts` — the `qwen-tts` package's internal
download path missed subdirectory configs on cold-start.

Expected: ~3.4 GB MPS delta → total ~6.6 GB with Whisper gone (sequential
loads). End-to-end measurement is committed alongside the
`samples/week1/output-es.mp4` artifact.

### End-to-end (run_dub sequential load/unload)

Pipeline structure (per-stage device routing):

| Stage | Model | Device | Dtype | Peak RSS |
|---|---|---|---|---|
| ASR | `Systran/faster-whisper-large-v3` | CPU int8 | int8 | **3.86 GB** |
| MT  | `facebook/m2m100_418M` (default)  | MPS fp32 | fp32 | **3.86 GB** (no delta over ASR baseline) |
| MT  | `google/madlad400-3b-mt` (opt-in) | CPU (forced) | bf16 | ~6 GB resident (measured pending) |
| TTS | `Qwen/Qwen3-TTS-12Hz-1.7B-Base`   | MPS fp32 | fp32 | **6.63 GB** |

All three stages run sequentially; the 6.63 GB TTS peak is the pipeline ceiling.
The M1 8 GB target has ~1.4 GB headroom under the measured peak.

**`memory_pressure -l critical` validation (2026-05-19):** pipeline ran to completion
(11/11 segments) under sustained critical memory pressure. Peak RSS dropped to 5.93 GB
(macOS page compression reduces the RSS figure under pressure; real unified memory
allocation is stable). Exit code 0. **8 GB fit confirmed.**

Sequential load/unload guarantees only one stage's resident memory at
any time; `_unload_if_other` + `gc.collect()` + `torch.mps.empty_cache()`
runs between every stage transition.

### Locked decisions

- **TTS:** `Qwen/Qwen3-TTS-12Hz-1.7B-Base` via `qwen-tts>=0.1.1` (Apache-2.0).
- **MT (default, 8 GB-safe):** `facebook/m2m100_418M` (MIT).
- **MT (opt-in, ≥16 GB):** `google/madlad400-3b-mt` (Apache-2.0), CPU bf16.
- **ASR:** `Systran/faster-whisper-large-v3` (MIT), CPU int8 on macOS.
- **Rejected:** Voicebox (no permissive release), NLLB (CC-BY-NC),
  MADLAD-1B (does not exist on a permissive license).
