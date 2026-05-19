"""Engine A/B benchmark for OPE-19.

Loads each candidate TTS engine, runs a fixed prompt through it, records
peak RSS + wall time + on-disk weight size, and writes a table to stdout.
Intended for the week-1 spike write-up.

Usage:

    python sidecar/bench_tts.py [--engines qwen3-tts,qwen3-tts-quantized]

The script intentionally does not call into the sidecar's stdin/stdout
protocol — it shares the loader code via module-level imports so we
benchmark the same code path that ships, with no IPC overhead in the
numbers.
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from pathlib import Path
from typing import List

# Make `sidecar/main.py` importable as `lc_sidecar` for the loaders.
SIDE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SIDE_DIR))
import main as lc_sidecar  # noqa: E402


def peak_rss_mb() -> float:
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


def bench_one(engine: str, device: str) -> dict:
    print(f"\n=== benchmarking {engine!r} on {device!r} ===", file=sys.stderr)
    t0 = time.time()
    rss_before = peak_rss_mb()
    try:
        lc_sidecar._load_tts(engine, device)
    except lc_sidecar.SidecarError as exc:
        return {
            "engine": engine,
            "device": device,
            "status": "load_failed",
            "error": str(exc),
        }
    load_secs = time.time() - t0
    rss_after = peak_rss_mb()
    return {
        "engine": engine,
        "device": device,
        "status": "loaded",
        "load_secs": round(load_secs, 2),
        "rss_before_mb": round(rss_before, 1),
        "rss_after_mb": round(rss_after, 1),
        "delta_mb": round(rss_after - rss_before, 1),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engines",
        default="qwen3-tts,qwen3-tts-quantized",
        help="comma-separated engines to benchmark",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="torch device hint (mps|cuda|cpu)",
    )
    args = parser.parse_args(argv)

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    device = lc_sidecar._resolve_torch_device(args.device)
    print(f"resolved device: {device}", file=sys.stderr)

    results = [bench_one(e, device) for e in engines]

    print()
    print("| engine | status | load (s) | RSS delta (MB) | error |")
    print("| --- | --- | --- | --- | --- |")
    for r in results:
        err = r.get("error", "") or ""
        if len(err) > 80:
            err = err[:77] + "…"
        print(
            f"| {r['engine']} | {r['status']} | "
            f"{r.get('load_secs', '—')} | {r.get('delta_mb', '—')} | {err} |"
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
