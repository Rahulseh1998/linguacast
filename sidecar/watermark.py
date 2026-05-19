"""linguacast perceptual audio watermark — clean-room implementation.

This is the load-bearing safety claim of LinguaCast v0 (per OPE-13). The
watermark says "this audio came from LinguaCast" even after ID3/XMP
metadata is stripped, and survives a YouTube-style 1080p H.264 + AAC
re-encode at ≥80% on the survival corpus.

Algorithm
---------
Patchwork-style spread-spectrum watermark on the log-magnitude STFT of
the mid-band (1 kHz – 6 kHz). The mid-band is chosen because:

  * AAC at 192 kbps keeps the spectral envelope across this band — it is
    where speech sits, so the psychoacoustic quantizer cannot afford to
    aggressively prune it.
  * The band is above the most aggressive temporal-masking effects below
    1 kHz and below the AAC hard cutoff that often sits around 16 kHz
    for the bitrates we target.

For every bit of payload, we:

  1. Derive a per-bit pseudo-random partition of the mid-band STFT bins
     into two equal-size sets A and B, keyed by ``HMAC-SHA256(KEY, bit_index)``.
  2. For each STFT frame inside the bit-window (default 25 frames ≈ 1.07 s
     at sr=24 kHz, hop=1024), perturb the log-magnitude:
         bit=1 → log|X(k∈A)| += α   and   log|X(k∈B)| -= α
         bit=0 → log|X(k∈A)| -= α   and   log|X(k∈B)| += α
     where α is the embedding strength (default 0.12 nepers, ~1 dB).

  3. Inverse STFT with the same Hann window and OLA reconstruction.

Detection is blind (no original needed) — for each candidate bit-window
we compute the same A/B partition statistic and threshold the soft bits.
A 16-bit sync prefix is searched across a small range of STFT-frame
offsets to handle the small phase/time shifts that re-encoding can
introduce. The 64-bit payload format is:

    sync (16 bits, constant 0xACE1) | watermark_id (32) | version (8) | crc8 (8)

The whole 64-bit code is repeated to fill the available audio so that
re-encoded clips can vote across copies before reporting "detected".

Reference: Bender, Gruhl, Morimoto, Lu — "Techniques for data hiding",
IBM Systems Journal 35(3/4), 1996. The Patchwork section. Filed >27 years
ago, so the underlying technique is now in the public domain. This
implementation is clean-room from the algorithm description; no third-
party watermark code was used.

License
-------
Apache-2.0 (same as the rest of LinguaCast). The module depends only on
numpy (BSD-3-Clause) and scipy (BSD-3-Clause).
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
from scipy.signal import istft as _scipy_istft, stft as _scipy_stft

# ---- Constants -----------------------------------------------------------

# Static project key. The watermark is *not* a secret — anyone can verify
# its presence, that is the whole point. The key is fixed so any
# linguacast build can recover any other build's watermark.
LINGUACAST_WM_KEY: bytes = b"linguacast.v0.watermark"

SYNC_BITS: int = 16
SYNC_VALUE: int = 0xACE1  # 1010 1100 1110 0001
ID_BITS: int = 32
VERSION_BITS: int = 8
CRC_BITS: int = 8
PAYLOAD_BITS: int = SYNC_BITS + ID_BITS + VERSION_BITS + CRC_BITS  # 64

# STFT parameters. 2048 / 512 @ 24 kHz gives ~85 ms windows / 21 ms hop,
# which lines up well with AAC's ~20–40 ms framing. The window has to be
# short enough that a 64-bit payload fits inside a 60-second canonical
# clip with room for several repeats (majority-voted on detect).
FRAME_SIZE: int = 2048
HOP: int = 512
EMBED_BAND_LO_HZ: float = 1000.0
EMBED_BAND_HI_HZ: float = 6000.0

# Default embedding strength. Tuned on TTS output: α=0.12 ≈ 1 dB swing
# per partition, which is below the ear's just-noticeable difference on
# speech mid-band and above AAC-192 quantization noise. The survival
# corpus on OPE-13 measures the floor.
DEFAULT_ALPHA: float = 0.12

# Number of STFT frames per payload bit. 10 frames × 64 bits = 640 frames
# per pass = ~13.7 s per pass at sr=24 kHz, hop=512. A 60 s canonical clip
# fits ~4 passes — enough redundancy to majority-vote past AAC bit flips.
DEFAULT_BIT_FRAMES: int = 10

# Sync search range, in STFT frames. A 1 kHz pitch shift or a 50 ms time
# offset from an AAC re-encode is well under this.
SYNC_SEARCH_FRAMES: int = 12


# ---- CRC-8 (poly 0x07 — same as the ITU "CRC-8/I-CODE" / ATM-HEC) -----

def _crc8(data_bits: List[int]) -> int:
    """8-bit CRC over a list of bits (MSB-first). Polynomial 0x07."""
    crc = 0
    for bit in data_bits:
        crc ^= (bit & 1) << 7
        for _ in range(1):
            crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


def _bits_from_int(value: int, width: int) -> List[int]:
    return [(value >> (width - 1 - i)) & 1 for i in range(width)]


def _int_from_bits(bits: Iterable[int], width: int) -> int:
    value = 0
    for b in list(bits)[:width]:
        value = (value << 1) | (b & 1)
    return value


# ---- Payload pack / unpack ---------------------------------------------

@dataclass
class WatermarkPayload:
    """Decoded payload — 32-bit watermark id, version byte, CRC ok flag."""

    watermark_id: int        # 32-bit
    version: int             # 8-bit (major<<4 | minor, free schema)
    crc_ok: bool

    @property
    def watermark_id_hex(self) -> str:
        return f"{self.watermark_id:08x}"


def pack_payload(watermark_id: int, version: int = 0x10) -> List[int]:
    """Build the 64-bit payload bit list."""
    if not (0 <= watermark_id < 1 << 32):
        raise ValueError("watermark_id must be a 32-bit unsigned int")
    if not (0 <= version < 1 << 8):
        raise ValueError("version must be an 8-bit unsigned int")
    sync = _bits_from_int(SYNC_VALUE, SYNC_BITS)
    idb = _bits_from_int(watermark_id, ID_BITS)
    verb = _bits_from_int(version, VERSION_BITS)
    crc = _crc8(idb + verb)
    crcb = _bits_from_int(crc, CRC_BITS)
    return sync + idb + verb + crcb


def unpack_payload(bits: List[int]) -> Optional[WatermarkPayload]:
    if len(bits) < PAYLOAD_BITS:
        return None
    sync = _int_from_bits(bits[:SYNC_BITS], SYNC_BITS)
    if sync != SYNC_VALUE:
        return None
    off = SYNC_BITS
    idb = bits[off:off + ID_BITS]
    off += ID_BITS
    verb = bits[off:off + VERSION_BITS]
    off += VERSION_BITS
    crcb = bits[off:off + CRC_BITS]
    crc_calc = _crc8(idb + verb)
    crc_recv = _int_from_bits(crcb, CRC_BITS)
    return WatermarkPayload(
        watermark_id=_int_from_bits(idb, ID_BITS),
        version=_int_from_bits(verb, VERSION_BITS),
        crc_ok=(crc_calc == crc_recv),
    )


# ---- STFT helpers ------------------------------------------------------

def _hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def _stft(audio: np.ndarray, frame_size: int, hop: int) -> np.ndarray:
    """Mono float32 → complex64 [n_frames, n_bins].

    Delegates to scipy.signal.stft with a Hann window and
    ``boundary='zeros'``/``padded=True`` so the same boundary handling is
    reproduced by ``_istft`` for COLA reconstruction. The returned shape
    is ``[n_frames, n_bins]`` (transposed from scipy's ``[n_bins, n_frames]``).
    """
    if audio.ndim != 1:
        raise ValueError("watermark works on mono audio only")
    a = audio.astype(np.float32, copy=False)
    _, _, Z = _scipy_stft(
        a,
        nperseg=frame_size,
        noverlap=frame_size - hop,
        window="hann",
        return_onesided=True,
        boundary="zeros",
        padded=True,
        scaling="spectrum",
    )
    return Z.T.astype(np.complex64, copy=False)


def _istft(frames: np.ndarray, frame_size: int, hop: int, length: int) -> np.ndarray:
    """Inverse STFT — COLA Hann OLA via scipy. Returns ``length`` samples."""
    if frames.shape[0] == 0:
        return np.zeros(length, dtype=np.float32)
    _, x = _scipy_istft(
        frames.T,
        nperseg=frame_size,
        noverlap=frame_size - hop,
        window="hann",
        input_onesided=True,
        boundary=True,
        scaling="spectrum",
    )
    x = x.astype(np.float32, copy=False)
    if len(x) >= length:
        return x[:length]
    pad = np.zeros(length - len(x), dtype=np.float32)
    return np.concatenate([x, pad]).astype(np.float32, copy=False)


def _band_bins(sr: int, frame_size: int) -> np.ndarray:
    """Return the FFT-bin indices that fall in [EMBED_BAND_LO_HZ, EMBED_BAND_HI_HZ]."""
    n_bins = frame_size // 2 + 1
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sr)
    mask = (freqs >= EMBED_BAND_LO_HZ) & (freqs <= EMBED_BAND_HI_HZ)
    bins = np.nonzero(mask)[0]
    if len(bins) < 32:
        raise ValueError(
            f"sample rate {sr} Hz is too low for the {EMBED_BAND_LO_HZ}-{EMBED_BAND_HI_HZ} "
            "Hz watermark band — bumps required"
        )
    return bins


def _partition(bit_index: int, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pseudo-random equal partition of bins into (A, B), keyed by bit_index."""
    seed_bytes = hmac.new(LINGUACAST_WM_KEY, bit_index.to_bytes(4, "big"), hashlib.sha256).digest()
    seed = int.from_bytes(seed_bytes[:4], "big")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(bins))
    half = len(bins) // 2
    a = bins[perm[:half]]
    b = bins[perm[half:half * 2]]
    return a, b


# ---- Embed / detect ----------------------------------------------------

def embed(
    audio: np.ndarray,
    sr: int,
    watermark_id: int,
    version: int = 0x10,
    alpha: float = DEFAULT_ALPHA,
    bit_frames: int = DEFAULT_BIT_FRAMES,
) -> np.ndarray:
    """Embed a 64-bit payload across the whole audio.

    Returns a float32 ndarray of the same length as the input.
    """
    if audio.ndim != 1:
        raise ValueError("watermark works on mono audio")
    audio = audio.astype(np.float32, copy=True)
    length = len(audio)
    if length < FRAME_SIZE * 2:
        # Audio too short for any embedding; pass through.
        return audio
    payload = pack_payload(watermark_id, version)

    frames = _stft(audio, FRAME_SIZE, HOP)
    n_frames = frames.shape[0]
    bins = _band_bins(sr, FRAME_SIZE)

    eps = 1e-10
    # Apply the per-frame patchwork perturbation in the log-magnitude
    # domain. We keep phase intact so phase-coding artefacts don't leak
    # into perceived audio.
    mag = np.abs(frames).astype(np.float32)
    phase = np.angle(frames)

    # Repeat the payload across the available frames.
    bits_total_frames = n_frames
    bits_per_pass = PAYLOAD_BITS * bit_frames
    if bits_per_pass <= 0:
        return audio

    for frame_i in range(n_frames):
        bit_pos_in_pass = (frame_i // bit_frames) % PAYLOAD_BITS
        bit = payload[bit_pos_in_pass]
        a_bins, b_bins = _partition(bit_pos_in_pass, bins)
        # Patchwork: bit=1 boosts A and attenuates B; bit=0 the reverse.
        sign = 1.0 if bit == 1 else -1.0
        log_a = np.log(mag[frame_i, a_bins] + eps) + sign * alpha
        log_b = np.log(mag[frame_i, b_bins] + eps) - sign * alpha
        mag[frame_i, a_bins] = np.exp(log_a)
        mag[frame_i, b_bins] = np.exp(log_b)

    new_frames = (mag * np.exp(1j * phase)).astype(np.complex64)
    out = _istft(new_frames, FRAME_SIZE, HOP, length)

    # Gentle clip guard — TTS output sits well below ±1 so this is mostly
    # a safety net for adversarial inputs.
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 0.99:
        out = (out * (0.99 / peak)).astype(np.float32)
    return out


@dataclass
class WatermarkDetection:
    detected: bool
    confidence: float                 # 0..1, normalised per-bit margin
    payload: Optional[WatermarkPayload]
    bit_offset_frames: int
    repeats_voted: int

    def to_dict(self) -> dict:
        out = {
            "detected": self.detected,
            "confidence": round(self.confidence, 4),
            "bit_offset_frames": self.bit_offset_frames,
            "repeats_voted": self.repeats_voted,
        }
        if self.payload is not None:
            out["watermark_id"] = self.payload.watermark_id_hex
            out["version"] = self.payload.version
            out["crc_ok"] = self.payload.crc_ok
        return out


def _frame_bit_statistic(mag_frame: np.ndarray, a_bins: np.ndarray, b_bins: np.ndarray) -> float:
    """Soft bit estimate: mean log-mag(A) - mean log-mag(B). Positive ⇒ bit 1."""
    eps = 1e-10
    la = np.mean(np.log(mag_frame[a_bins] + eps))
    lb = np.mean(np.log(mag_frame[b_bins] + eps))
    return float(la - lb)


def detect(
    audio: np.ndarray,
    sr: int,
    bit_frames: int = DEFAULT_BIT_FRAMES,
    confidence_threshold: float = 0.55,
) -> WatermarkDetection:
    """Blind detection across the whole audio.

    Returns the best alignment over a small sync search range, with a
    confidence score derived from the per-bit decision margin averaged
    across voted repeats.
    """
    if audio.ndim != 1:
        audio = audio.mean(axis=-1).astype(np.float32) if audio.ndim == 2 else audio
    audio = audio.astype(np.float32, copy=False)
    if len(audio) < FRAME_SIZE * 2:
        return WatermarkDetection(False, 0.0, None, 0, 0)

    frames = _stft(audio, FRAME_SIZE, HOP)
    mag = np.abs(frames).astype(np.float32)
    n_frames = frames.shape[0]
    bins = _band_bins(sr, FRAME_SIZE)

    # Precompute the per-frame, per-bit statistic — n_frames × PAYLOAD_BITS.
    # Cheap because each is a few hundred bins.
    bit_partitions = [_partition(i, bins) for i in range(PAYLOAD_BITS)]
    stats = np.zeros((n_frames, PAYLOAD_BITS), dtype=np.float32)
    for bi, (a_bins, b_bins) in enumerate(bit_partitions):
        # vectorised across frames
        la = np.log(mag[:, a_bins] + 1e-10).mean(axis=1)
        lb = np.log(mag[:, b_bins] + 1e-10).mean(axis=1)
        stats[:, bi] = la - lb

    sync_bits = _bits_from_int(SYNC_VALUE, SYNC_BITS)
    best = WatermarkDetection(False, 0.0, None, 0, 0)

    # Search a small range of starting offsets in STFT-frame units. Rank
    # candidates as (crc_ok, confidence) so a valid CRC beats a slightly
    # noisier alignment that happens to score marginally higher.
    def _score(d: WatermarkDetection) -> Tuple[int, float]:
        crc = 1 if (d.payload is not None and d.payload.crc_ok) else 0
        return crc, d.confidence

    for offset in range(-SYNC_SEARCH_FRAMES, SYNC_SEARCH_FRAMES + 1):
        det = _decode_at_offset(stats, sync_bits, bit_frames, offset, confidence_threshold)
        if _score(det) > _score(best):
            best = det
        if (
            best.detected
            and best.payload is not None
            and best.payload.crc_ok
            and best.confidence > 0.85
        ):
            # Strong enough — stop searching to keep verify fast.
            break
    return best


def _decode_at_offset(
    stats: np.ndarray,
    sync_bits: List[int],
    bit_frames: int,
    offset: int,
    confidence_threshold: float,
) -> WatermarkDetection:
    n_frames, n_bits = stats.shape
    bits_per_pass = PAYLOAD_BITS * bit_frames
    start = offset
    # Skip negative starts cleanly.
    if start + bits_per_pass > n_frames:
        return WatermarkDetection(False, 0.0, None, offset, 0)

    # Vote each bit-position's signed margin across all repeats present.
    margins = np.zeros(PAYLOAD_BITS, dtype=np.float64)
    counts = np.zeros(PAYLOAD_BITS, dtype=np.int32)
    repeats = 0
    pos = max(start, 0)
    while pos + bits_per_pass <= n_frames:
        for bi in range(PAYLOAD_BITS):
            frame_lo = pos + bi * bit_frames
            frame_hi = frame_lo + bit_frames
            if frame_hi > n_frames:
                break
            margins[bi] += float(stats[frame_lo:frame_hi, bi].mean())
            counts[bi] += 1
        pos += bits_per_pass
        repeats += 1
    if repeats == 0:
        return WatermarkDetection(False, 0.0, None, offset, 0)

    avg_margins = margins / np.maximum(counts, 1)
    hard_bits = [1 if m > 0 else 0 for m in avg_margins]
    # Check sync.
    sync_match = sum(1 for i in range(SYNC_BITS) if hard_bits[i] == sync_bits[i])
    if sync_match < SYNC_BITS - 2:
        # >2 sync errors — almost certainly not aligned.
        return WatermarkDetection(False, 0.0, None, offset, repeats)
    payload = unpack_payload(hard_bits)
    # Confidence: fraction of bit decisions whose margin magnitude
    # exceeds a noise floor (median absolute margin × small factor).
    mag_margins = np.abs(avg_margins)
    floor = float(np.median(mag_margins))
    confident_bits = float(np.mean(mag_margins > floor * 1.1))
    sync_score = sync_match / SYNC_BITS
    confidence = float(0.5 * sync_score + 0.5 * confident_bits)
    detected = (
        payload is not None
        and (payload.crc_ok or sync_score == 1.0)
        and confidence >= confidence_threshold
    )
    return WatermarkDetection(
        detected=detected,
        confidence=confidence,
        payload=payload,
        bit_offset_frames=offset,
        repeats_voted=repeats,
    )


# ---- Random watermark-id helper ----------------------------------------

def random_watermark_id() -> int:
    """A fresh 32-bit watermark id — caller persists this alongside metadata."""
    import secrets
    return int.from_bytes(secrets.token_bytes(4), "big")
