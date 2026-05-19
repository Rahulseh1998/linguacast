"""linguacast Python sidecar.

Process-orchestrated from the Rust CLI. One JSON request per line on stdin,
one JSON response per line on stdout. Stderr is for human-readable progress.

Sequential load/unload architecture
-----------------------------------
Per the OPE-19 launch-hook constraint (≤6 GB resident on an M1 8 GB Air,
measured under `memory_pressure -l critical`), this sidecar holds **one
model resident at a time**. Each stage (`transcribe`, `translate`, `tts`,
`run_dub`) loads its model, runs to completion, then explicitly frees it
with `del` + `gc.collect()` + `torch.mps.empty_cache()` before the next
stage is allowed to allocate. There is no "keep models hot" path. The
per-stage peak RSS is captured (via `psutil`) and returned in each
response so the orchestrator can surface it to the user without scraping
stderr.

Engine roster
-------------
- ASR: ``faster-whisper`` 1.x running Whisper-large-v3 (MIT, BSD-3-Clause).
  CTranslate2 does not expose Metal, so macOS gets CPU int8 (~2× realtime
  on a 60s clip, ~2.5 GB resident — about a quarter of openai-whisper
  on MPS). CUDA gets fp16.
- MT: ``transformers`` + MADLAD-400-3B-MT (Apache-2.0). bf16 on MPS/CUDA,
  fp32 on CPU. Prompt prefix ``<2es>``.
- TTS: ``qwen-tts`` 0.1.x wrapping ``Qwen/Qwen3-TTS-12Hz-1.7B-Base``
  (Apache-2.0). fp32 on MPS (fp16 trips the multinomial sampler),
  bf16 on CUDA, fp32 on CPU. SDPA attention everywhere
  (flash-attn is CUDA-only).

The ``-CustomVoice`` variant is kept as a documented fallback in
``docs/engine-decision.md`` — the OPE-4 track validated ``-Base`` against
the real ``qwen-tts`` package API, so that's the one wired here.
"""

from __future__ import annotations

import gc
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SIDECAR_VERSION = "0.2.0-dev"

CACHE_ROOT = Path(
    os.environ.get("LINGUACAST_CACHE_DIR")
    or (Path.home() / ".cache" / "linguacast")
)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE_ROOT / "hf"))


def log(msg: str) -> None:
    """Human progress goes to stderr; stdout is reserved for JSON."""
    print(f"[sidecar] {msg}", file=sys.stderr, flush=True)


def emit(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def emit_error(message: str, *, recoverable: bool = False) -> None:
    emit({"kind": "error", "message": message, "recoverable": recoverable})


class SidecarError(RuntimeError):
    """Raised inside an op handler — converted to a structured error response."""


# --- Memory accounting ----------------------------------------------------
#
# We want a single number per stage that the launch-hook conversation can
# point at: "Whisper peaked at X MB, MADLAD at Y MB, Qwen3-TTS at Z MB". We
# use psutil for current RSS (cross-platform) and the resource module for
# the historical peak (RUSAGE_SELF/ru_maxrss is in kilobytes on macOS, bytes
# on Linux — the helper normalises to MB).

_psutil = None


def _psutil_module():
    global _psutil
    if _psutil is None:
        try:
            import psutil  # pyright: ignore
        except Exception as exc:  # pragma: no cover
            raise SidecarError(
                f"psutil is required for memory instrumentation: {exc}"
            ) from exc
        _psutil = psutil
    return _psutil


def current_rss_mb() -> float:
    proc = _psutil_module().Process(os.getpid())
    return proc.memory_info().rss / (1024 * 1024)


def peak_rss_mb_since_reset(reset_baseline: Optional[float] = None) -> float:
    """Return the historical RSS high-water mark for this process, in MB.

    ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` is monotonic, so we record
    a baseline at the top of each stage and report ``max(current, peak)``
    minus the baseline isn't meaningful (peak only ever grows). What we
    really want is the absolute peak across the whole pipeline run, which
    is what the orchestrator's `--memory-pressure-test` mode checks against
    the 8 GB ceiling. The `reset_baseline` arg lets callers normalise if
    they want delta peaks for a specific stage; the default returns the
    absolute peak.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On macOS ru_maxrss is bytes; on Linux it's kilobytes. Detect by
    # platform rather than guessing from magnitude.
    if sys.platform == "darwin":
        peak_mb = raw / (1024 * 1024)
    else:
        peak_mb = raw / 1024
    if reset_baseline is not None:
        return max(0.0, peak_mb - reset_baseline)
    return peak_mb


# --- Sequential load/unload registry --------------------------------------

# A single slot for "the model currently loaded". Anything in here gets
# freed when a different stage is requested. We deliberately do NOT keep a
# per-stage cache — the launch-hook constraint is one resident model at a
# time, full stop.

_torch = None
_loaded_stage: Optional[str] = None
_loaded_handle: Any = None


def _torch_module():
    global _torch
    if _torch is None:
        log("importing torch (this is the slow one)…")
        import torch  # pyright: ignore

        _torch = torch
    return _torch


def _empty_accel_cache() -> None:
    """Best-effort accelerator-cache flush after a `del`."""
    torch = _torch_module()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            # Older torches don't have mps.empty_cache; fine.
            pass


def _unload_if_other(stage: str) -> None:
    """Sequential load/unload entry-point. Frees the resident model if it
    isn't the one ``stage`` needs. Always called before a stage loads.
    """
    global _loaded_stage, _loaded_handle
    if _loaded_stage is None or _loaded_stage == stage:
        return
    log(f"unloading {_loaded_stage} before loading {stage}…")
    _loaded_handle = None
    _loaded_stage = None
    gc.collect()
    _empty_accel_cache()


def _unload_all() -> None:
    """Free whatever is resident, used at the end of `run_dub`."""
    global _loaded_stage, _loaded_handle
    if _loaded_stage is None:
        return
    log(f"unloading {_loaded_stage} (pipeline complete)…")
    _loaded_handle = None
    _loaded_stage = None
    gc.collect()
    _empty_accel_cache()


def _set_loaded(stage: str, handle: Any) -> None:
    global _loaded_stage, _loaded_handle
    _loaded_stage = stage
    _loaded_handle = handle


def _resolve_torch_device(requested: str) -> str:
    """Translate Rust's device hint into something torch will accept.

    Returns 'mps' | 'cuda' | 'cpu'. Falls back transparently if the
    requested accelerator isn't available — the Rust side surfaces the
    actual device chosen back to the user.
    """
    torch = _torch_module()
    if requested == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested in ("mps", "cuda"):
        log(f"requested device {requested!r} not available; falling back to cpu")
    return "cpu"


# --- ASR: faster-whisper --------------------------------------------------

def _whisper_compute(torch_device: str) -> Tuple[str, str]:
    """faster-whisper accepts only cuda/cpu (CTranslate2). Pick compute_type.

    CTranslate2 has no Metal backend, so MPS users get CPU int8 — which is
    still ~2× realtime on a 60s clip on M-series and uses ~2.5 GB instead
    of ~10 GB resident.
    """
    if torch_device == "cuda":
        return "cuda", "float16"
    return "cpu", "int8"


def _load_whisper(size: str) -> Any:
    """Load Whisper via faster-whisper (CTranslate2). Sequential-load aware."""
    _unload_if_other("whisper")
    if _loaded_stage == "whisper":
        return _loaded_handle
    log(f"loading whisper-{size} via faster-whisper…")
    from faster_whisper import WhisperModel  # pyright: ignore

    torch_device = _resolve_torch_device(
        "cuda" if _torch_module().cuda.is_available() else "cpu"
    )
    device, compute_type = _whisper_compute(torch_device)
    model = WhisperModel(
        size,
        device=device,
        compute_type=compute_type,
        download_root=str(CACHE_ROOT / "models" / "whisper"),
    )
    _set_loaded("whisper", model)
    return model


# --- MT: MADLAD-400 --------------------------------------------------------

def _load_madlad(size: str, device: str) -> Tuple[Any, Any]:
    """Load MADLAD-400 (3B or 1B) on `device`. Sequential-load aware."""
    _unload_if_other("madlad")
    if _loaded_stage == "madlad":
        return _loaded_handle
    model_id = MADLAD_MODELS.get(size)
    if model_id is None:
        raise SidecarError(f"unknown madlad size: {size!r} (use '3b' or '1b')")
    log(f"loading {model_id} on {device}…")
    import torch  # pyright: ignore
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # pyright: ignore

    if device == "cuda":
        dtype = torch.bfloat16
    elif device == "mps":
        # MADLAD on MPS: bf16 works on M2/M3, but M1 has had intermittent
        # NaN reports with bf16. fp16 is the safe default for the launch
        # hook and matches what faster-whisper uses on the same hardware.
        dtype = torch.float16
    else:
        dtype = torch.float32
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    handle = (mdl, tok, device)
    _set_loaded("madlad", handle)
    return handle


# MADLAD-400 family (all Apache-2.0). The CTO ack pre-approved `madlad-1b`
# as the fallback if 3B doesn't fit, but Google's only sub-3B public MADLAD
# release is on a non-Apache license (MADLAD-Base is research-only). For
# the launch hook we keep `3b` as the only registry entry; if 3B OOMs the
# real fallback is to switch MT families entirely, and that escalates back
# to the CTO per the rubric in `docs/engine-decision.md`.
MADLAD_MODELS = {
    "3b": "google/madlad400-3b-mt",
    "10b": "google/madlad400-10b-mt",
}


# --- TTS: qwen-tts wrapping Qwen3-TTS-12Hz-1.7B-Base ----------------------

# ISO 639-1 → English names that Qwen3-TTS's voice-clone path expects.
QWEN_LANG_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}


def _qwen_lang_name(iso: str) -> str:
    return QWEN_LANG_NAMES.get(iso.lower(), "Auto")


def _load_qwen_tts(size: str, device: str) -> Any:
    """Load Qwen3-TTS-12Hz-{size}-Base via the qwen-tts package."""
    _unload_if_other("tts")
    if _loaded_stage == "tts":
        return _loaded_handle
    if size not in ("0.6B", "1.7B"):
        raise SidecarError(
            f"LINGUACAST_TTS_SIZE={size!r} is not supported; use '0.6B' or '1.7B'"
        )
    model_id = f"Qwen/Qwen3-TTS-12Hz-{size}-Base"
    log(f"loading {model_id} on {device} (this downloads ~3.4 GB on first run)…")
    import torch  # pyright: ignore

    try:
        from qwen_tts import Qwen3TTSModel  # pyright: ignore
    except Exception as exc:
        raise SidecarError(
            f"qwen-tts package is required for TTS but isn't importable: {exc}. "
            f"Install with `pip install qwen-tts`."
        ) from exc

    # MPS + voice_clone needs fp32 — fp16 trips the multinomial sampler at
    # decode time. CUDA prefers bf16. CPU stays fp32.
    if device == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        dtype=dtype,
        attn_implementation="sdpa",  # flash-attn is CUDA-only
        cache_dir=str(CACHE_ROOT / "models" / "qwen3-tts"),
    )
    handle = (model, size)
    _set_loaded("tts", handle)
    return handle


# --- Op handlers -----------------------------------------------------------


def op_hello(_: Dict[str, Any]) -> Dict[str, Any]:
    """Quick handshake. Imports torch and reports the best-available device."""
    torch = _torch_module()
    has_mps = bool(
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    )
    has_cuda = bool(torch.cuda.is_available())
    if has_mps:
        device = "mps"
    elif has_cuda:
        device = "cuda"
    else:
        device = "cpu"
    return {
        "kind": "hello",
        "version": SIDECAR_VERSION,
        "torch_device": device,
        "torch_version": torch.__version__,
    }


def op_pull(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Pre-download all model weights and return a sized inventory.

    Used by `linguacast pull` so the warm-cache TTW measurement is honest:
    cold downloads stay separate from the warm-cache run.
    """
    asr = payload.get("asr", "large-v3")
    mt_size = payload.get("mt", "3b")
    tts_size = payload.get("tts", "1.7B")
    log(f"pull: asr={asr} mt={mt_size} tts={tts_size}")

    # Whisper download. faster-whisper download is implicit in WhisperModel(),
    # so we load + unload as the simplest way to force the fetch.
    _ = _load_whisper(asr)
    _unload_if_other("__none__")  # force unload

    # MADLAD download. transformers caches under HF_HOME.
    device = _resolve_torch_device("cpu")  # CPU just to force the fetch
    _ = _load_madlad(mt_size, device)
    _unload_if_other("__none__")

    # Qwen3-TTS download. qwen-tts uses HF caching as well.
    _ = _load_qwen_tts(tts_size, device)
    _unload_if_other("__none__")

    cache_root = str(CACHE_ROOT)
    return {
        "kind": "pull",
        "cache_root": cache_root,
        "models": {
            "asr": f"whisper-{asr}",
            "mt": MADLAD_MODELS[mt_size],
            "tts": f"Qwen/Qwen3-TTS-12Hz-{tts_size}-Base",
        },
    }


def op_transcribe(payload: Dict[str, Any]) -> Dict[str, Any]:
    """ASR via faster-whisper. Loads → runs → unloads."""
    audio_path = Path(payload["audio_path"])
    asr_size = payload.get("asr", "large-v3")
    if not audio_path.exists():
        raise SidecarError(f"audio file not found: {audio_path}")

    stage_t0 = time.time()
    model = _load_whisper(asr_size)
    log(f"transcribing {audio_path.name}…")
    t0 = time.time()
    segments_iter, info = model.transcribe(
        str(audio_path),
        language="en",
        vad_filter=True,
        word_timestamps=False,
        beam_size=5,
    )
    segments: List[Dict[str, Any]] = []
    for s in segments_iter:
        text = (s.text or "").strip()
        if not text:
            continue
        segments.append(
            {"start": float(s.start or 0.0), "end": float(s.end or 0.0), "text": text}
        )
    transcribe_secs = time.time() - t0
    log(f"  whisper took {transcribe_secs:.2f}s · {len(segments)} segments")

    peak_mb = peak_rss_mb_since_reset()
    rss_mb = current_rss_mb()
    # Free the model now so the next op starts from a clean slate.
    _unload_if_other("__none__")

    return {
        "kind": "transcribe",
        "language": info.language or "en",
        "segments": segments,
        "stage_seconds": time.time() - stage_t0,
        "transcribe_seconds": transcribe_secs,
        "peak_rss_mb": peak_mb,
        "current_rss_mb": rss_mb,
        "model": f"whisper-{asr_size}",
    }


def op_translate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """MT via MADLAD-400. Loads → runs → unloads."""
    segments_in: List[Dict[str, Any]] = payload["segments"]
    source_lang: str = payload.get("source_lang", "en")
    target_lang: str = payload["target_lang"]
    mt_size = payload.get("mt", "3b")
    device = _resolve_torch_device(payload.get("device", "cpu"))

    if not segments_in:
        return {"kind": "translate", "segments": [], "model": MADLAD_MODELS[mt_size]}

    stage_t0 = time.time()
    model, tok, device_used = _load_madlad(mt_size, device)

    log(f"translating {len(segments_in)} segments {source_lang} → {target_lang}…")
    target_token = f"<2{target_lang}>"
    out_segments: List[Dict[str, Any]] = []
    t0 = time.time()
    for seg in segments_in:
        prompt = f"{target_token} {seg['text']}"
        inputs = tok(prompt, return_tensors="pt").to(device_used)
        with _torch_module().inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=4,
            )
        translated = tok.decode(generated[0], skip_special_tokens=True).strip()
        out_segments.append(
            {"start": seg["start"], "end": seg["end"], "text": translated}
        )
    translate_secs = time.time() - t0
    log(f"  madlad took {translate_secs:.2f}s")

    peak_mb = peak_rss_mb_since_reset()
    rss_mb = current_rss_mb()
    _unload_if_other("__none__")

    return {
        "kind": "translate",
        "segments": out_segments,
        "stage_seconds": time.time() - stage_t0,
        "translate_seconds": translate_secs,
        "peak_rss_mb": peak_mb,
        "current_rss_mb": rss_mb,
        "model": MADLAD_MODELS[mt_size],
    }


def op_tts(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Voice-clone TTS via Qwen3-TTS-12Hz-1.7B-Base.

    Synthesises each translated segment with the source clip as the speaker
    reference, places each clip at the source timestamp on a target-duration
    track, writes PCM_16 WAV. Loads → runs → unloads.
    """
    segments: List[Dict[str, Any]] = payload["segments"]
    reference: Path = Path(payload["reference_audio_path"])
    target_lang: str = payload["target_lang"]
    out_audio: Path = Path(payload["out_audio_path"])
    target_duration: float = float(payload.get("target_duration_sec", 0.0))
    ref_text: str = (payload.get("ref_text") or "").strip()
    tts_size: str = payload.get("tts_size", os.environ.get("LINGUACAST_TTS_SIZE", "1.7B"))
    device = _resolve_torch_device(payload.get("device", "cpu"))

    if not reference.exists():
        raise SidecarError(f"reference audio not found: {reference}")
    if not segments:
        raise SidecarError("nothing to synthesize — translation returned 0 segments")

    stage_t0 = time.time()
    model, size_loaded = _load_qwen_tts(tts_size, device)

    import numpy as np  # pyright: ignore
    import soundfile as sf  # pyright: ignore

    out_sr = 24000  # qwen-tts emits 24 kHz mono float
    if target_duration <= 0:
        # Fallback: span the latest segment end + 1s headroom.
        target_duration = max((s["end"] for s in segments), default=1.0) + 1.0
    track = np.zeros(int(max(target_duration, 1.0) * out_sr), dtype=np.float32)
    ref_audio = str(reference)
    ref_text = ref_text or "Speaker reference clip."
    lang_name = _qwen_lang_name(target_lang)

    t0 = time.time()
    rendered = 0
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=lang_name,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        audio = _to_mono_float32(wavs[0])
        if sr != out_sr:
            audio = _linear_resample(audio, sr, out_sr)
        start_idx = int(float(seg["start"]) * out_sr)
        if start_idx >= len(track):
            continue
        end_idx = min(start_idx + len(audio), len(track))
        track[start_idx:end_idx] = audio[: end_idx - start_idx]
        rendered += 1
    synth_secs = time.time() - t0
    log(f"  qwen3-tts rendered {rendered}/{len(segments)} segments in {synth_secs:.2f}s")

    out_audio.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_audio), track, out_sr, subtype="PCM_16")

    peak_mb = peak_rss_mb_since_reset()
    rss_mb = current_rss_mb()
    _unload_if_other("__none__")

    return {
        "kind": "tts",
        "out_audio_path": str(out_audio),
        "duration_sec": float(target_duration),
        "sample_rate": out_sr,
        "segments_rendered": rendered,
        "stage_seconds": time.time() - stage_t0,
        "synth_seconds": synth_secs,
        "peak_rss_mb": peak_mb,
        "current_rss_mb": rss_mb,
        "model": f"Qwen/Qwen3-TTS-12Hz-{size_loaded}-Base",
    }


def op_run_dub(payload: Dict[str, Any]) -> Dict[str, Any]:
    """End-to-end dub in a single IPC call.

    Convenient for the orchestrator: one process spawn, sequential
    load/unload between Whisper → MADLAD → Qwen3-TTS, single response
    with the aggregated per-stage RSS table.
    """
    audio_path = Path(payload["audio_path"])
    reference_audio = Path(payload.get("reference_audio_path", audio_path))
    target_lang: str = payload["target_lang"]
    out_audio: Path = Path(payload["out_audio_path"])
    target_duration: float = float(payload.get("target_duration_sec", 0.0))
    asr_size = payload.get("asr", "large-v3")
    mt_size = payload.get("mt", "3b")
    tts_size = payload.get("tts_size", os.environ.get("LINGUACAST_TTS_SIZE", "1.7B"))
    device = _resolve_torch_device(payload.get("device", "cpu"))

    stages: List[Dict[str, Any]] = []

    # Stage 1: transcribe
    asr_resp = op_transcribe({"audio_path": str(audio_path), "asr": asr_size})
    stages.append(
        {
            "name": "asr",
            "model": asr_resp["model"],
            "stage_seconds": asr_resp["stage_seconds"],
            "peak_rss_mb": asr_resp["peak_rss_mb"],
        }
    )

    segments = asr_resp["segments"]
    ref_text = " ".join(s["text"] for s in segments).strip()

    # Stage 2: translate
    mt_resp = op_translate(
        {
            "segments": segments,
            "source_lang": asr_resp["language"],
            "target_lang": target_lang,
            "mt": mt_size,
            "device": device,
        }
    )
    stages.append(
        {
            "name": "mt",
            "model": mt_resp["model"],
            "stage_seconds": mt_resp["stage_seconds"],
            "peak_rss_mb": mt_resp["peak_rss_mb"],
        }
    )

    # Stage 3: tts
    tts_resp = op_tts(
        {
            "segments": mt_resp["segments"],
            "reference_audio_path": str(reference_audio),
            "target_lang": target_lang,
            "out_audio_path": str(out_audio),
            "target_duration_sec": target_duration,
            "ref_text": ref_text,
            "tts_size": tts_size,
            "device": device,
        }
    )
    stages.append(
        {
            "name": "tts",
            "model": tts_resp["model"],
            "stage_seconds": tts_resp["stage_seconds"],
            "peak_rss_mb": tts_resp["peak_rss_mb"],
        }
    )

    _unload_all()

    return {
        "kind": "run_dub",
        "out_audio_path": tts_resp["out_audio_path"],
        "duration_sec": tts_resp["duration_sec"],
        "sample_rate": tts_resp["sample_rate"],
        "language": asr_resp["language"],
        "target_lang": target_lang,
        "segments": len(mt_resp["segments"]),
        "segments_rendered": tts_resp["segments_rendered"],
        "stages": stages,
        "peak_rss_mb": max(s["peak_rss_mb"] for s in stages),
    }


def _to_mono_float32(x):
    import numpy as np  # pyright: ignore

    a = np.asarray(x)
    if a.ndim == 2:
        a = a.mean(axis=0) if a.shape[0] < a.shape[1] else a.mean(axis=1)
    return a.astype(np.float32, copy=False)


def _linear_resample(audio, sr_in: int, sr_out: int):
    """Linear resample. Good enough for 24 kHz drift; ffmpeg muxes back to
    AAC at 48 kHz anyway, so the band-limit story is on ffmpeg's side."""
    import numpy as np  # pyright: ignore

    if sr_in == sr_out:
        return audio
    ratio = sr_out / sr_in
    n_out = int(len(audio) * ratio)
    x_in = np.linspace(0.0, 1.0, num=len(audio), endpoint=False, dtype=np.float64)
    x_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float64)
    return np.interp(x_out, x_in, audio).astype(np.float32, copy=False)


OPS = {
    "hello": op_hello,
    "pull": op_pull,
    "transcribe": op_transcribe,
    "translate": op_translate,
    "tts": op_tts,
    "run_dub": op_run_dub,
}


def main() -> int:
    log(f"sidecar {SIDECAR_VERSION} ready · cache={CACHE_ROOT}")
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            emit_error(f"bad json from orchestrator: {exc}", recoverable=True)
            continue
        op = payload.get("op")
        handler = OPS.get(op)
        if handler is None:
            emit_error(f"unknown op: {op!r}", recoverable=True)
            continue
        try:
            emit(handler(payload))
        except SidecarError as exc:
            emit_error(str(exc), recoverable=True)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            emit_error(f"{type(exc).__name__}: {exc}", recoverable=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
