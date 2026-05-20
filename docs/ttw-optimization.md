# TTW Optimization Log (OPE-44)

## Baseline (week-1, commit `b0a6bd6`)

- **Wall time:** 210s for a 58s clip, single-language (EN→ES), M1 MPS warm cache
- **Bottleneck:** Qwen3-TTS synthesis ~166s (79% of total)
- **Other stages:** Whisper ~29s, M2M-100 ~8s, mux <2s
- **Peak RSS:** 6.63 GB

Goal: ≤180s warm-cache single-language path on M1 8 GB.

---

## Optimization 1: Fix spurious auto-pull on macOS (2026-05-19)

**Root cause:** `pipeline.rs::cache_looks_empty()` checks `~/Library/Caches/linguacast/hf`
(the macOS `BaseDirs::cache_dir()` path), but the sidecar always writes to
`~/.cache/linguacast/hf` (XDG path, unconditional). On macOS, the two paths are
different, so `cache_looks_empty()` returned `true` on every invocation, triggering
a full `op_pull` call even when all models were already cached.

**Fix:** `cache_looks_empty()` now also checks `~/.cache/linguacast/` (the
sidecar's XDG path) before declaring the cache empty. Warm-cache runs on macOS
no longer trigger the pull handshake (~15-30s saved per invocation).

**Files:** `crates/linguacast/src/pipeline.rs` (`cache_looks_empty`)

---

## Optimization 2: Batched TTS generation (2026-05-19)

**Opportunity:** Qwen3-TTS `generate_voice_clone()` accepts `text: List[str]`,
`language: List[str]`, `ref_audio: List[...]`, `ref_text: List[...]` for batched
inference. The sidecar was calling it once per segment (sequential).

**Fix:** Group segments into batches of 4 (configurable via `LINGUACAST_TTS_BATCH`
env var) and call `generate_voice_clone()` with list inputs. The batch size was
chosen conservatively for 8 GB M1 headroom; larger batches may be faster on
≥16 GB hosts.

**Expected gain:** Model-level amortization of the cross-attention + kv-cache
overhead across segments. Expected 15-30% reduction in synthesis time based on
typical transformer batching curves, putting warm-cache TTW at ~145-175s.

**Note:** Actual measured gain pending a timed run post-implementation.
Set `LINGUACAST_TTS_BATCH=1` to revert to sequential for debugging.

**Files:** `sidecar/main.py` (`op_tts`)

---

## Combined expected TTW (estimate)

| Change | Saved (estimate) |
| --- | --- |
| Fix macOS auto-pull | ~20s |
| Batch TTS (batch=4) | ~25-50s |
| **Total** | **~45-70s** |
| **New estimated TTW** | **~140-165s** |

Target ≤180s is achievable. Actual measurement on next full benchmark run.

---

## Rejected / deferred optimizations

- **CTranslate2 for Qwen3-TTS:** Not supported. Qwen3-TTS uses HuggingFace
  Transformers internals not yet ported to CTranslate2 (verified 2026-05-19).
- **INT8/Q4 quant:** No Apache-2.0 quantized Qwen3-TTS 1.7B checkpoint on HF
  Hub as of 2026-05-19. The 0.6B variant is available as a fallback (`--tts-size 0.6B`)
  but has noticeable voice quality degradation on non-Spanish langs.
- **ASR+MT prefetch while TTS renders:** Architecture requires sequential
  load/unload for 8 GB fit; adding a prefetch thread would require holding two
  models in memory simultaneously. Deferred to week 3 when we revisit the 8 GB
  ceiling for larger hosts.
