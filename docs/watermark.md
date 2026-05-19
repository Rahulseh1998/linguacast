# LinguaCast watermark & provenance — OPE-13

The launch-blocker safety story: **every voice-cloned output ships with
a perceptual audio watermark and namespaced container-metadata flags so
anyone can run `linguacast verify <file>` and learn whether a file came
out of LinguaCast — even if the metadata has been stripped.**

Status: landed for v0 (week-3 launch-blocker per
[OPE-13](/OPE/issues/OPE-13)).

## TL;DR — what every output gets

1. **Audio watermark.** A 64-bit payload (`16-bit sync · 32-bit
   watermark_id · 8-bit version · 8-bit CRC-8`) embedded in the
   1 kHz–6 kHz band of the synthesized speech using a patchwork-style
   spread-spectrum scheme. Inaudible (perturbation is ~1.8 dB per
   partition, below the speech-mid-band JND); recoverable after a
   YouTube-style 1080p H.264 + AAC re-encode.
2. **Metadata flags.** ID3-style fields written as MP4 udta atoms via
   ffmpeg: `linguacast_consent_hash`, `linguacast_consent_timestamp`,
   `linguacast_consent_signer`, `linguacast_version`,
   `linguacast_watermark_id`, `linguacast_watermark_algo`. Plus a
   parseable `comment` mirror so non-namespace-aware players can still
   surface them.
3. **Verifier.** `linguacast verify <file>` runs the detector against
   the file's audio track, reads the metadata, and reports whether the
   recovered watermark id matches the metadata claim. The audio
   watermark is the *load-bearing* claim; the metadata is convenience
   and survives only until someone runs `ffmpeg -map_metadata -1`.

## Algorithm

Patchwork spread-spectrum on the log-magnitude STFT of the mid-band.
Reference: Bender, Gruhl, Morimoto, Lu —
*"Techniques for data hiding"*, IBM Systems Journal 35(3/4), 1996.
The Patchwork section. The technique was published >27 years ago, so
the underlying method is now in the public domain; this implementation
is clean-room from the algorithm description.

### Parameters (`sidecar/watermark.py`)

| Constant | Value | Why |
|---|---|---|
| `FRAME_SIZE` | 2048 samples | 85 ms windows @ 24 kHz — matches AAC framing |
| `HOP` | 512 samples | 75% overlap; COLA Hann reconstruction |
| `EMBED_BAND_LO_HZ` | 1000 | Above the most aggressive low-band perceptual masking |
| `EMBED_BAND_HI_HZ` | 6000 | Below the AAC hard cutoff for 128–192 kbps profiles |
| `DEFAULT_ALPHA` | 0.20 | ~1.8 dB swing per partition — below the speech-mid-band JND |
| `DEFAULT_BIT_FRAMES` | 8 | 512 frames per 64-bit pass = ~10.9 s/pass — 5+ passes in 60 s |

### Payload schema (64 bits, MSB-first)

```
| 16 sync (0xACE1) | 32 watermark_id | 8 version | 8 CRC-8 (0x07) |
```

- `watermark_id` is `u32::from_be_bytes(sha256(consent_hash_hex)[..4])`.
  Deterministic per consent record, so the bit-payload cross-checks
  the metadata-claimed `linguacast_watermark_id`.
- `version` is currently `0x10` (major=1, minor=0 of the watermark
  format itself — not the same as `linguacast_version` in metadata).
- `CRC-8` covers the id+version bits. Used as a sanity check on
  decode; a CRC pass means the bit-payload survived intact.

### Embed/detect flow

- **Embed.** STFT → for each frame, look up its `bit_pos = (frame //
  bit_frames) % 64`, generate the pseudo-random partition A/B for that
  bit (keyed by `HMAC-SHA256(KEY, bit_index)`), perturb
  `log|X(k∈A)| += α` and `log|X(k∈B)| -= α` for bit=1 (signs flipped
  for bit=0). Phase is left untouched to avoid phase-coding audibility.
  Inverse STFT with COLA Hann → write WAV.
- **Detect.** STFT the candidate. For each candidate sync offset
  `[-12, +12]` frames, compute the per-bit signed margin
  `mean log|X(k∈A)| - mean log|X(k∈B)|` averaged across all repeats
  of the payload that fit in the audio. Hard-decode to bits, check
  sync, attempt to unpack id/version/CRC, score (CRC-OK > confidence).

The 24-frame sync search window absorbs the small time/phase
offsets that AAC/Opus encode-decode pipelines introduce.

## Survival corpus

Tested with 5 samples × 12 transforms = **60 trials**. Each watermarked
master is `wav-embed → AAC-mux to MP4` then each transform is applied
to that master before extraction-and-detect. Acceptance: ≥80% on the
1080p H.264 + AAC re-encode and metadata-stripped paths.

Reproduce with:

```bash
sidecar/.venv/bin/python scripts/watermark_survival.py \
  --samples samples/week1/input.mp4 samples/week1/output-es.mp4 \
            /tmp/wm-clips/input-30s.mp4 /tmp/wm-clips/input-mid30s.mp4 \
            /tmp/wm-clips/es-30s.mp4 \
  --corpus-dir /tmp/wm-survival
```

(See `scripts/watermark_survival.py` for the full transform list. The
raw per-trial CSV lives at `docs/watermark-survival.csv` — regenerate
on every change to the watermark module.)

| Transform | Detected | id-match | CRC-ok | Notes |
|---|---|---|---|---|
| `identity` | 100% | 80% | 80% | ffmpeg container round-trip (still goes audio→AAC@192k once) |
| `aac-128` | 100% | 80% | 80% | |
| `aac-192` | 100% | 80% | 80% | |
| `aac-256` | 100% | 80% | 80% | |
| **`youtube-1080p`** | **100%** | **80%** | **80%** | **OPE-13 acceptance criterion — passes ≥80% bar** |
| `youtube-720p` | 100% | 80% | 80% | Lower-end mobile pipeline |
| **`metadata-stripped`** | **100%** | **80%** | **80%** | **`-map_metadata -1` — proves the watermark, not metadata, is load-bearing** |
| `aac-64` | 100% | 60% | 60% | Aggressive quantization — id payload starts to flip |
| `mp3-128` | 100% | 100% | 100% | MP3 is friendlier than AAC at this rate |
| `opus-96` | 100% | 80% | 80% | WebRTC / Discord / many podcast pipelines |
| `volume-up-6db` | 100% | 80% | 80% | +6 dB gain before re-encode |
| `volume-down-6db` | 100% | 80% | 80% | -6 dB gain before re-encode |

### What "80%" means in practice

- **Detection is 100% on every transform**, including `metadata-stripped`
  and the very aggressive `aac-64`. So the binary claim "this is a
  LinguaCast output" survives universally on this corpus — that's the
  load-bearing safety claim.
- **id-match at 80%** means in 1 of 5 trials the recovered 32-bit id
  has a small number of bit flips (typically 1–3 of 32 bits) that the
  CRC catches. The bit-pattern is consistent across transforms within
  a sample, suggesting it's the underlying content of that sample
  pushing AAC to allocate quantization noise in a way the embedder
  didn't fully mask, rather than the codec being the variable.
- The one failing sample (`es-30s`) is a 30-second clip — at the lower
  end of the duration where only ~2 payload repeats fit. Longer clips
  (the canonical 60-second sample) hit 100% id-match.

The OPE-13 spec bar: *"If <80% survives, flag and decide whether to
ship and accept the gap or escalate to CTO before public alpha."* The
80% bar is met cleanly on every transform that matters for the
launch story (YouTube re-encode, metadata-strip, common codecs).
No escalation needed.

### Known gaps / v0 limitations

- Corpus is small (n=5). The launch-blocker bar is met but a larger
  corpus would tighten the confidence interval. Adding more samples
  is cheap — re-run `scripts/watermark_survival.py` with more
  `--samples`.
- `aac-64` is the only transform that drops below 80% on id-match.
  We don't ship any output at 64 kbps and the detection rate is still
  100%, so this is documented as a known gap, not a blocker.
- The watermark requires at least ~11 s of audio (one payload pass).
  Shorter clips will trigger the watermark's degraded-redundancy mode
  and may not be detectable. The CLI clip length is bounded by the
  source video, so a short input → short output → short watermark
  margin.
- The watermark embeds in the synthesized speech track, not the
  original-language audio. If someone takes the dubbed output, strips
  the audio, and replaces it with their own narration, the watermark
  is gone. This is by design — we only mark *our* synthesized voice,
  not arbitrary audio in the file.
- The video frames themselves are not watermarked. v0.2 if needed
  (per OPE-13 "Out of scope").

## License audit

The watermark module (`sidecar/watermark.py`) is Apache-2.0 (same as
the rest of LinguaCast) and depends only on numpy (BSD-3-Clause) and
scipy (BSD-3-Clause). No third-party watermark library was used —
every OSS perceptual-audio-watermark library surveyed at the time of
landing (audiowmark/GPL-2.0, WavMark/research-only,
silentcipher/research-only, AudioSeal/restricted) failed the
Apache-2.0-or-MIT floor. The Patchwork algorithm itself was published
in 1996 and is public-domain knowledge.

## Verifier UX

```
$ linguacast verify path/to/output.mp4

LinguaCast verify — path/to/output.mp4

Audio watermark
  detected      : yes
  confidence    : 0.703
  watermark id  : 4f022f7c
  version byte  : 0x10
  crc           : ok
  repeats voted : 4
  sample rate   : 24000 Hz
  audio duration: 58.33 s
  detect time   : 0.239 s
  algorithm     : patchwork-spread-spectrum-v1

Container metadata (ID3/XMP-style, may be stripped)
  linguacast_consent_hash      : 8a74278100a2…
  linguacast_consent_signer    : rahul@…
  linguacast_consent_timestamp : 2026-05-19T22:21:33Z
  linguacast_version           : 0.1.0-dev
  linguacast_watermark_algo    : patchwork-spread-spectrum-v1
  linguacast_watermark_id      : 4f022f7c
  …

Provenance: ✓ audio watermark id matches the linguacast_watermark_id in metadata.
```

`linguacast verify --json <file>` emits the same data as a JSON
document for downstream tooling. Both paths report all three cases:

- Watermark + matching metadata id → confirmed provenance.
- Watermark + missing/stripped metadata → still confirmed as LinguaCast.
- Watermark + mismatched metadata id → flagged as tampering/remux.
- No watermark → not a LinguaCast output (or audio was heavily reprocessed).

## What's intentionally out of scope (v0)

- Network/public verifier service. The CLI is local-only.
- Watermarking the *video* track (lip-sync frames). The launch story is
  about the *voice*; v0.2 can revisit.
- Removing the watermark is not adversarial-grade hardened. If someone
  bandpass-filters out 1 kHz–6 kHz, the watermark goes too — but so
  does most of the intelligible speech. Any adversary willing to do
  that has already destroyed the value of the dubbed audio.
- Public-key signing of the watermark. The watermark is meant to be
  *recoverable by anyone*, not authenticated, so plain HMAC-keyed
  partitions are sufficient for v0. Authentication (a separate signed
  manifest) is a fast-follow if downstream tooling asks for it.
