# Engine decision: Qwen3-TTS vs Voicebox on M1 8 GB

**Owner:** LinguaCoder · **Issue:** [OPE-19](https://example.invalid/OPE-19) · **Status:** In progress (week 1)

## Why this decision matters

The launch hook is *clone the repo → dubbed clip in 3 minutes on an M1*.
The TTS engine is the single biggest determinant of:

1. **Whether we fit memory** — Qwen3-TTS and Voicebox are both multi-GB
   models. The reference target is the M1 8 GB Air, which has ~6 GB free
   unified memory under realistic conditions. If neither engine fits, the
   launch hook breaks.
2. **Voice clone quality** — the demo's "wow" moment is hearing yourself
   in Spanish. If the cloned voice doesn't sound like the source, the
   product doesn't deliver on its core claim.
3. **Inference latency** — 60 seconds of audio in <3 minutes wall time
   means the TTS step has to be ≤90 seconds (the rest of the pipeline
   uses the remaining budget).

## Rubric

The decision is graded against five criteria, in priority order:

| # | Criterion | Pass bar | Weight |
| --- | --- | --- | --- |
| 1 | **License** — Apache-2.0 or MIT on the released weights | hard floor (binary) | reject if fails |
| 2 | **Fits M1 8 GB** — resident memory ≤6 GB during inference | hard floor (binary) | reject if fails after quantization |
| 3 | **Voice clone quality** — A/B blind listening test on the EN→ES sample, judged "the same speaker" by ≥3 of 5 listeners | soft (subjective) | high |
| 4 | **Latency** — ≤1.5× realtime on M1 MPS (so 60s of audio in ≤90s wall) | soft | high |
| 5 | **Stability** — runs to completion on the canonical 60s clip without OOM or NaN | hard | medium |

Voice quality is the tiebreaker if both engines pass the hard floors.

## Candidate snapshot

| Engine | License | Approx. params | Quantization story | Native voice clone |
| --- | --- | --- | --- | --- |
| Qwen3-TTS | Apache-2.0 (per HF model card) | ~7B (full) / 4B (quantized variant) | int4 / int8 via Qwen's own quantized release | Yes — zero-shot from a 3–10s reference clip |
| Voicebox (Meta) | **Under review** — public release license unclear; original paper says research-only | ~330M (base) / 2.5B (large) | n/a — already small enough | Yes — flow-matching, very high fidelity |

**Status of the Voicebox license check (as of 2026-05-19):** the original
2023 paper made the weights research-only. A 2024 community fork is
floating around but is not Meta-blessed. Until I can verify the released
weights I am integrating against are Apache-2.0 / MIT, the sidecar
refuses to load Voicebox — `--tts voicebox` returns a clear error. If the
license can be cleared we revisit. If not, Voicebox is out for v0 and
we'll evaluate alternative open TTS (XTTS, Parler-TTS, F5-TTS) in week 2.

## Test protocol

Sample: `samples/week1/input.mp4` — 60-second EN clip, single speaker.

For each engine:

1. Load the model with the requested device (MPS first, CPU fallback if
   load fails).
2. Record cold-start time (process spawn → first token / sample emitted).
3. Record peak resident memory (via `ps -o rss=` polled at 250 ms).
4. Run the EN→ES translation through the engine using the input clip as
   the voice reference.
5. Mux back over the original video; play and listen.
6. Repeat on simulated 8 GB ceiling: set `LINGUACAST_MAX_RSS_MB=6144` and
   re-run; abort if the process exceeds it.

Results land in this doc under the "Results" section below.

## Decision flow

```
[Qwen3-TTS full] — fits 8 GB?
  ├── yes → use it
  └── no  → [Qwen3-TTS quantized] — fits 8 GB?
              ├── yes → use it
              └── no  → [Voicebox] — license cleared AND fits 8 GB?
                          ├── yes → use it
                          └── no  → ESCALATE to CTO before spending >1 day
                                    on a workaround (per kickoff comment).
                                    Alternatives: XTTS-v2 (CPL but check),
                                    Parler-TTS (Apache-2.0), or F5-TTS
                                    (MIT). All week-2 territory.
```

## Results

> _Filled in during the spike. As of this commit, the rubric is set and
> the loaders are wired — the actual benchmark numbers go here once the
> deps are installed and the engine has run end-to-end on the sample._

### Cold load time

| Engine | Wall time | Peak RSS at load |
| --- | --- | --- |
| Qwen3-TTS full | _pending_ | _pending_ |
| Qwen3-TTS quantized | _pending_ | _pending_ |
| Voicebox | _blocked on license check_ | — |

### Inference (60s ES dub on canonical clip)

| Engine | Wall time | Peak RSS | Audio MOS (informal) | Voice similarity (informal) |
| --- | --- | --- | --- | --- |
| Qwen3-TTS full | _pending_ | _pending_ | _pending_ | _pending_ |
| Qwen3-TTS quantized | _pending_ | _pending_ | _pending_ | _pending_ |

### Recommendation

_To be filled in once both rows above are complete. Default-leaning
toward Qwen3-TTS quantized given the 8 GB ceiling, with full-precision
as the `--tts qwen3-tts` opt-in for machines with headroom._
