"""CI smoke-test sidecar. Speaks the JSON-per-line IPC protocol without
loading a single model. Used by `.github/workflows/ci.yml` to verify
the Rust orchestrator wiring on macos-arm64 / macos-x64 / linux-x64
runners without paying the ~10 GB model download cost.

Real-model behaviour lives in sidecar/main.py. This file deliberately
mirrors only the response *shapes* the Rust side parses; any model-quality
question must be answered by the real sidecar."""

from __future__ import annotations

import json
import os
import sys
import wave
from pathlib import Path


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def write_silent_wav(path: Path, duration_sec: float, sample_rate: int = 24000):
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = max(1, int(duration_sec * sample_rate))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * n_frames)


def handle(op, payload):
    if op == "hello":
        return {
            "kind": "hello",
            "version": "0.0.0-ci-mock",
            "torch_device": "cpu",
            "torch_version": "mock",
        }
    if op == "pull":
        return {
            "kind": "pull",
            "cache_root": os.environ.get(
                "LINGUACAST_CACHE_DIR", str(Path.home() / ".cache" / "linguacast")
            ),
            "models": {
                "asr": "mock-whisper",
                "mt": "mock-m2m100",
                "tts": "mock-qwen3-tts",
            },
        }
    if op == "transcribe":
        return {
            "kind": "transcribe",
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello"},
            ],
            "peak_rss_mb": 0.0,
        }
    if op == "translate":
        return {
            "kind": "translate",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hola"},
            ],
            "peak_rss_mb": 0.0,
        }
    if op == "tts":
        out_audio = Path(payload["out_audio_path"])
        write_silent_wav(out_audio, payload.get("target_duration_sec", 1.0))
        return {
            "kind": "tts",
            "out_audio_path": str(out_audio),
            "duration_sec": float(payload.get("target_duration_sec", 1.0)),
            "peak_rss_mb": 0.0,
        }
    if op == "run_dub":
        out_audio = Path(payload["out_audio_path"])
        duration = float(payload.get("target_duration_sec", 1.0))
        write_silent_wav(out_audio, duration)
        # Emit a couple of progress events so the orchestrator can drain them.
        emit({"kind": "progress", "stage": "asr", "phase": "infer"})
        emit({"kind": "progress", "stage": "mt", "phase": "infer", "current": 1, "total": 1})
        emit({"kind": "progress", "stage": "tts", "phase": "infer", "current": 1, "total": 1})
        return {
            "kind": "run_dub",
            "out_audio_path": str(out_audio),
            "duration_sec": duration,
            "sample_rate": 24000,
            "language": "en",
            "target_lang": payload.get("target_lang", "es"),
            "segments": 1,
            "segments_rendered": 1,
            "stages": [
                {"name": "asr", "model": "mock", "stage_seconds": 0.1, "peak_rss_mb": 0.0},
                {"name": "mt", "model": "mock", "stage_seconds": 0.1, "peak_rss_mb": 0.0},
                {"name": "tts", "model": "mock", "stage_seconds": 0.1, "peak_rss_mb": 0.0},
            ],
            "peak_rss_mb": 0.0,
        }
    return {"kind": "error", "message": f"mock sidecar does not implement op {op!r}", "recoverable": True}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            emit({"kind": "error", "message": f"bad json: {exc}", "recoverable": True})
            continue
        op = payload.get("op")
        emit(handle(op, payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
