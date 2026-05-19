#!/usr/bin/env python3
"""OPE-13 watermark survival corpus runner.

For each sample input MP4, this script:

  1. Builds a watermarked output (via the simulate script — full-pipeline
     dub is too slow to iterate; the watermark embedder is identical).
  2. Runs N transforms on the watermarked MP4:
       - identity (round-trip ffmpeg copy)
       - aac-128 re-encode (audio only)
       - aac-192 re-encode (audio only)
       - aac-256 re-encode (audio only)
       - youtube-1080p (full re-encode: H.264 + AAC, the OPE-13 acceptance test)
       - youtube-720p
       - stripped (metadata stripped via -map_metadata -1, watermark must
         still survive — this is the test that proves the watermark is
         load-bearing, not the metadata)
  3. For each transform, runs the sidecar verify op and records
     detected / crc_ok / watermark_id match.

Outputs a CSV report and a markdown summary on stdout. Tests are
intentionally deterministic per sample so re-runs reproduce numbers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sidecar"))

import numpy as np
import soundfile as sf

import watermark as wm  # noqa: E402  (sidecar local module)


# Each transform is `(name, ffmpeg-args-after-input, description)`. The
# `{ar}` placeholder is replaced with the original sample rate so we don't
# accidentally resample-test what is meant to be a codec-only test.
TRANSFORMS = [
    ("identity",
     ["-c:v", "copy", "-c:a", "copy"],
     "ffmpeg container round-trip, no re-encode."),

    ("aac-128",
     ["-c:v", "copy", "-c:a", "aac", "-b:a", "128k"],
     "AAC re-encode at 128 kbps (audio-only, video copy)."),

    ("aac-192",
     ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k"],
     "AAC re-encode at 192 kbps (audio-only)."),

    ("aac-256",
     ["-c:v", "copy", "-c:a", "aac", "-b:a", "256k"],
     "AAC re-encode at 256 kbps (audio-only)."),

    ("youtube-1080p",
     ["-c:v", "libx264", "-preset", "fast", "-crf", "23",
      "-vf", "scale=-2:1080",
      "-c:a", "aac", "-b:a", "192k"],
     "OPE-13 acceptance: 1080p H.264 + AAC re-encode."),

    ("youtube-720p",
     ["-c:v", "libx264", "-preset", "fast", "-crf", "23",
      "-vf", "scale=-2:720",
      "-c:a", "aac", "-b:a", "128k"],
     "720p H.264 + AAC re-encode (lower-end mobile pipeline)."),

    ("metadata-stripped",
     ["-c:v", "copy", "-c:a", "copy", "-map_metadata", "-1"],
     "All container metadata stripped — the watermark must still survive."),

    ("aac-64",
     ["-c:v", "copy", "-c:a", "aac", "-b:a", "64k"],
     "AAC re-encode at 64 kbps — the aggressive-quantization floor (mobile / podcast)."),

    ("mp3-128",
     ["-c:v", "copy", "-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp4"],
     "MP3 audio re-encode — used by some platforms / DAWs."),

    ("opus-96",
     ["-c:v", "copy", "-c:a", "libopus", "-b:a", "96k", "-strict", "experimental", "-f", "mp4"],
     "Opus re-encode — WebRTC / Discord / many podcast pipelines."),

    ("volume-up-6db",
     ["-c:v", "copy", "-af", "volume=2.0", "-c:a", "aac", "-b:a", "192k"],
     "+6 dB gain before AAC re-encode (loudness-normalisation upstream)."),

    ("volume-down-6db",
     ["-c:v", "copy", "-af", "volume=0.5", "-c:a", "aac", "-b:a", "192k"],
     "-6 dB gain before AAC re-encode (quiet-mix upstream)."),
]


def ffmpeg(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def make_watermarked_master(source_mp4: Path, work: Path, consent_hash: str, output_id: str) -> tuple[Path, int]:
    """Build a watermarked MP4 master.

    Uses the canonical pipeline metadata layout. The watermark id is
    deterministically derived from the consent hash (high 32 bits of
    SHA-256(consent_hash)) so this matches what the live dub pipeline
    emits in pipeline.rs::watermark_id_from_consent.
    """
    digest = hashlib.sha256(consent_hash.encode()).digest()
    wid = int.from_bytes(digest[:4], "big")
    wid_hex = f"{wid:08x}"

    wav_in = work / f"{output_id}-in.wav"
    wav_out = work / f"{output_id}-out.wav"
    mp4_out = work / f"{output_id}-master.mp4"

    ffmpeg([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source_mp4),
        "-vn", "-ac", "1", "-ar", "24000", "-f", "wav",
        str(wav_in),
    ])
    audio, sr = sf.read(str(wav_in), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1).astype(np.float32)
    wm_audio = wm.embed(audio, sr=sr, watermark_id=wid)
    sf.write(str(wav_out), wm_audio, sr, subtype="PCM_16")

    meta_pairs = [
        ("comment", f"linguacast:consent_hash={consent_hash};lang=es;version=0.1.0-dev;watermark_id={wid_hex}"),
        ("linguacast_consent_hash", consent_hash),
        ("linguacast_version", "0.1.0-dev"),
        ("linguacast_watermark_id", wid_hex),
        ("linguacast_watermark_algo", "patchwork-spread-spectrum-v1"),
    ]
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source_mp4),
        "-i", str(wav_out),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart+use_metadata_tags",
    ]
    for k, v in meta_pairs:
        cmd += ["-metadata", f"{k}={v}"]
    cmd += [str(mp4_out)]
    ffmpeg(cmd)

    return mp4_out, wid


def detect_in_mp4(mp4_path: Path, work: Path) -> dict:
    """Extract audio @ 24k mono and run the watermark detector."""
    wav = work / (mp4_path.stem + ".24k.wav")
    ffmpeg([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(mp4_path),
        "-vn", "-ac", "1", "-ar", "24000", "-f", "wav",
        str(wav),
    ])
    audio, sr = sf.read(str(wav), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1).astype(np.float32)
    det = wm.detect(audio, sr=sr)
    return det.to_dict()


@dataclass
class TransformResult:
    sample: str
    transform: str
    detected: bool
    watermark_id: Optional[str]
    expected_id: str
    id_match: bool
    crc_ok: Optional[bool]
    confidence: float
    repeats_voted: int


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", required=True, help="source MP4(s)")
    ap.add_argument(
        "--out-csv",
        default=str(ROOT / "docs" / "watermark-survival.csv"),
        help="CSV report path",
    )
    ap.add_argument(
        "--corpus-dir",
        default=None,
        help="If set, keep the transformed MP4s here for manual listening.",
    )
    args = ap.parse_args()

    work_root = Path(args.corpus_dir) if args.corpus_dir else Path(tempfile.mkdtemp(prefix="wm-survival-"))
    work_root.mkdir(parents=True, exist_ok=True)
    print(f"work dir: {work_root}", file=sys.stderr)

    results: List[TransformResult] = []
    per_transform: Dict[str, Dict[str, int]] = {
        t[0]: {"trials": 0, "detected": 0, "id_match": 0, "crc_ok": 0}
        for t in TRANSFORMS
    }

    for src in args.samples:
        src_path = Path(src)
        if not src_path.exists():
            print(f"  skip (missing): {src_path}", file=sys.stderr)
            continue
        sample_name = src_path.stem
        # Use the file's own SHA-256 as the consent hash for repeatability.
        with open(src_path, "rb") as f:
            consent_hash = hashlib.sha256(f.read()).hexdigest()
        master, wid = make_watermarked_master(src_path, work_root, consent_hash, sample_name)
        wid_hex = f"{wid:08x}"
        print(f"sample={sample_name} consent_hash[:12]={consent_hash[:12]} expected_id={wid_hex}", file=sys.stderr)

        for name, args_after_input, _desc in TRANSFORMS:
            transformed = work_root / f"{sample_name}.{name}.mp4"
            try:
                ffmpeg([
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(master),
                    *args_after_input,
                    "-movflags", "+faststart",
                    str(transformed),
                ])
                det = detect_in_mp4(transformed, work_root)
            except subprocess.CalledProcessError as e:
                print(f"  {name}: ffmpeg failed: {e}", file=sys.stderr)
                continue

            recovered = det.get("watermark_id")
            id_match = (recovered is not None) and (recovered.lower() == wid_hex.lower())
            crc_ok = det.get("crc_ok")
            row = TransformResult(
                sample=sample_name,
                transform=name,
                detected=bool(det.get("detected", False)),
                watermark_id=recovered,
                expected_id=wid_hex,
                id_match=id_match,
                crc_ok=crc_ok if isinstance(crc_ok, bool) else None,
                confidence=float(det.get("confidence", 0.0)),
                repeats_voted=int(det.get("repeats_voted", 0)),
            )
            results.append(row)
            per_transform[name]["trials"] += 1
            if row.detected:
                per_transform[name]["detected"] += 1
            if row.id_match:
                per_transform[name]["id_match"] += 1
            if row.crc_ok:
                per_transform[name]["crc_ok"] += 1
            print(
                f"  {name:18s} detected={row.detected} id={row.watermark_id} "
                f"id_match={row.id_match} crc={row.crc_ok} conf={row.confidence:.3f}",
                file=sys.stderr,
            )

    # Write CSV.
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    print(f"wrote {out_csv}", file=sys.stderr)

    # Stdout markdown summary.
    print()
    print("## OPE-13 watermark survival corpus")
    print()
    print(f"- Samples: {len(set(r.sample for r in results))}")
    print(f"- Trials per transform: {per_transform[TRANSFORMS[0][0]]['trials']}")
    print(f"- Algorithm: patchwork-spread-spectrum-v1 (alpha={wm.DEFAULT_ALPHA}, bit_frames={wm.DEFAULT_BIT_FRAMES})")
    print()
    print("| Transform | Detected | id-match | CRC-ok |")
    print("|---|---|---|---|")
    for name, _args, _desc in TRANSFORMS:
        s = per_transform[name]
        n = s["trials"] or 1
        print(f"| `{name}` | {s['detected']}/{s['trials']} ({100*s['detected']/n:.0f}%) "
              f"| {s['id_match']}/{s['trials']} ({100*s['id_match']/n:.0f}%) "
              f"| {s['crc_ok']}/{s['trials']} ({100*s['crc_ok']/n:.0f}%) |")
    print()
    print("**Transforms in detail**")
    for name, _args, desc in TRANSFORMS:
        print(f"- `{name}`: {desc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
