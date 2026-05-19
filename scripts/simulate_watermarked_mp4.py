#!/usr/bin/env python3
"""Build a *simulated* watermarked LinguaCast MP4 from an existing output.

Used to exercise the `linguacast verify` integration without spinning the
full ASR→MT→TTS dub pipeline. Workflow:

  1. ffmpeg-extract the audio from `--input` as 24 kHz mono WAV.
  2. Embed the OPE-13 watermark with the given id (or one derived from
     a consent_hash so the cross-check matches a real consent record).
  3. ffmpeg-mux the watermarked WAV back over the source video, writing
     the same provenance metadata (linguacast_consent_hash / version /
     watermark_id / etc.) the real pipeline writes.

Output MP4 is byte-similar to what the real pipeline emits — the only
thing that differs is the audio content itself, which is taken from an
existing dub rather than re-synthesized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sidecar"))

import numpy as np
import soundfile as sf

import watermark as wm  # noqa: E402  (sidecar local module)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="MP4 to take audio+video from")
    ap.add_argument("--out", required=True, help="watermarked MP4 output path")
    ap.add_argument(
        "--consent-hash",
        default=None,
        help="If set, derive watermark_id = SHA-256(consent_hash)[:4]. Defaults to a fixed id.",
    )
    ap.add_argument(
        "--consent-record",
        default=None,
        help="Path to a consent record JSON (optional). Provides linguacast_version etc.",
    )
    ap.add_argument(
        "--watermark-id-hex",
        default=None,
        help="Override watermark id (32-bit hex). Wins over --consent-hash.",
    )
    args = ap.parse_args()

    if args.watermark_id_hex:
        wid = int(args.watermark_id_hex, 16) & 0xFFFFFFFF
        consent_hash = args.consent_hash or "0" * 64
    elif args.consent_hash:
        digest = hashlib.sha256(args.consent_hash.encode()).digest()
        wid = int.from_bytes(digest[:4], "big")
        consent_hash = args.consent_hash
    else:
        wid = 0xDECAFC0E
        consent_hash = "0" * 64

    wid_hex = f"{wid:08x}"

    if args.consent_record and Path(args.consent_record).exists():
        rec = json.loads(Path(args.consent_record).read_text())
        consent_hash = rec.get("audio_sha256", consent_hash)
        digest = hashlib.sha256(consent_hash.encode()).digest()
        wid = int.from_bytes(digest[:4], "big")
        wid_hex = f"{wid:08x}"
        version = rec.get("linguacast_version", "0.1.0-dev")
        signer = rec.get("signed_by", "synthetic@linguacast")
        timestamp = rec.get("timestamp_iso", "2026-05-19T00:00:00Z")
    else:
        version = "0.1.0-dev"
        signer = "synthetic@linguacast"
        timestamp = "2026-05-19T00:00:00Z"

    print(f"target watermark id: {wid_hex}  (consent_hash[:12]={consent_hash[:12]})", flush=True)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        wav_in = td / "in.wav"
        wav_out = td / "out.wav"

        # 1. Extract 24k mono WAV.
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(args.input),
                "-vn", "-ac", "1", "-ar", "24000", "-f", "wav",
                str(wav_in),
            ],
            check=True,
        )
        audio, sr = sf.read(str(wav_in), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1).astype(np.float32)

        # 2. Embed watermark.
        wm_audio = wm.embed(audio, sr=sr, watermark_id=wid)
        sf.write(str(wav_out), wm_audio, sr, subtype="PCM_16")

        # 3. Re-mux over source video with metadata.
        meta_pairs = [
            ("comment", f"linguacast:consent_hash={consent_hash};consent_ts={timestamp};signer={signer};lang=es;version={version};watermark_id={wid_hex}"),
            ("linguacast_consent_hash", consent_hash),
            ("linguacast_consent_timestamp", timestamp),
            ("linguacast_consent_signer", signer),
            ("linguacast_version", version),
            ("linguacast_watermark_id", wid_hex),
            ("linguacast_watermark_algo", "patchwork-spread-spectrum-v1"),
        ]
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(args.input),
            "-i", str(wav_out),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart+use_metadata_tags",
        ]
        for k, v in meta_pairs:
            cmd += ["-metadata", f"{k}={v}"]
        cmd += [str(args.out)]
        subprocess.run(cmd, check=True)

    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
