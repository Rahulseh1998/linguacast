"""Smoke / round-trip / survival tests for watermark.py.

Run from the sidecar dir under the venv:
    .venv/bin/python test_watermark.py

The survival test requires ffmpeg on PATH.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import watermark as wm


def _synth_speech_like(seconds: float, sr: int = 24000, seed: int = 1) -> np.ndarray:
    """Pseudo-speech: voiced pulse train + formant filter + noise.

    Not realistic, but spectrally rich in the mid-band where the
    watermark sits — far enough from pure tones that detection cannot
    cheat on a degenerate input.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    f0 = 110.0 + 5.0 * np.sin(2 * np.pi * 0.3 * t)
    pulses = (np.modf(np.cumsum(f0) / sr)[0] < 0.05).astype(np.float32)
    formant_lp = np.zeros(n, dtype=np.float32)
    alpha = 0.97
    acc = 0.0
    for i, x in enumerate(pulses):
        acc = alpha * acc + (1 - alpha) * x
        formant_lp[i] = acc
    formant = formant_lp + 0.3 * np.sin(2 * np.pi * 1200 * t) * pulses
    noise = 0.02 * rng.standard_normal(n).astype(np.float32)
    out = 0.4 * formant + noise
    return out.astype(np.float32)


def test_round_trip_clean() -> bool:
    sr = 24000
    audio = _synth_speech_like(20.0, sr=sr, seed=2)
    wid = 0x1234ABCD
    embedded = wm.embed(audio, sr=sr, watermark_id=wid)
    delta = embedded - audio
    rms = float(np.sqrt(np.mean(delta ** 2)))
    audio_rms = float(np.sqrt(np.mean(audio ** 2)))
    snr_db = 20.0 * np.log10(audio_rms / max(rms, 1e-9))
    det = wm.detect(embedded, sr=sr)
    print(
        f"  clean: detected={det.detected} conf={det.confidence:.3f} "
        f"id={det.payload.watermark_id_hex if det.payload else None} "
        f"crc_ok={det.payload.crc_ok if det.payload else None} "
        f"snr={snr_db:.1f}dB"
    )
    return (
        det.detected
        and det.payload is not None
        and det.payload.watermark_id == wid
        and det.payload.crc_ok
        and snr_db > 18.0  # watermark must stay well below the signal
    )


def test_clean_no_false_positive() -> bool:
    sr = 24000
    audio = _synth_speech_like(20.0, sr=sr, seed=99)
    det = wm.detect(audio, sr=sr)
    print(
        f"  no-wm: detected={det.detected} conf={det.confidence:.3f} "
        f"id={det.payload.watermark_id_hex if det.payload else None}"
    )
    return not det.detected


def test_aac_survival() -> bool:
    sr = 24000
    audio = _synth_speech_like(30.0, sr=sr, seed=7)
    wid = 0xCAFEBABE
    embedded = wm.embed(audio, sr=sr, watermark_id=wid)
    with tempfile.TemporaryDirectory() as td:
        in_wav = Path(td) / "in.wav"
        aac_m4a = Path(td) / "out.m4a"
        round_wav = Path(td) / "round.wav"
        sf.write(in_wav, embedded, sr, subtype="PCM_16")
        # AAC encode at 192 kbps then decode back — the YouTube-like audio path.
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(in_wav), "-c:a", "aac", "-b:a", "192k", str(aac_m4a)],
            check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(aac_m4a), "-ar", str(sr), "-ac", "1", str(round_wav)],
            check=True,
        )
        round_audio, _ = sf.read(round_wav, dtype="float32")
        det = wm.detect(round_audio, sr=sr)
        print(
            f"  aac-192: detected={det.detected} conf={det.confidence:.3f} "
            f"id={det.payload.watermark_id_hex if det.payload else None} "
            f"crc_ok={det.payload.crc_ok if det.payload else None}"
        )
        return (
            det.detected
            and det.payload is not None
            and det.payload.watermark_id == wid
            and det.payload.crc_ok
        )


def main() -> int:
    print("watermark.py smoke tests")
    ok = True
    for name, fn in [
        ("round-trip clean", test_round_trip_clean),
        ("no false positive", test_clean_no_false_positive),
        ("aac-192 survival", test_aac_survival),
    ]:
        print(f"- {name}")
        passed = False
        try:
            passed = fn()
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL: {type(e).__name__}: {e}")
        if not passed:
            ok = False
            print("  FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
