"""linguacast Python sidecar.

Process-orchestrated from the Rust CLI. One JSON request per line on stdin,
one JSON response per line on stdout. Stderr is for human-readable progress.

Why a sidecar (and not PyO3)? Because Whisper / MADLAD / Qwen3-TTS already
have first-class Python implementations, and shipping them as a subprocess
keeps the Rust side a thin orchestrator with a single, testable IPC seam.
We follow the same packaging pattern as Voicebox's official release: the
Rust binary is the entrypoint and the sidecar is lazy-installed/lazy-loaded.

The sidecar does NOT load models at startup. It loads on first use of each
stage, so `--help` / handshake stay snappy and the smoke test for missing
deps is fast.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

SIDECAR_VERSION = "0.1.0-dev"

# Cache root for models + intermediate downloads. Hugging Face will plant
# things under HF_HOME if we set it, which keeps everything reproducible and
# avoids polluting the user's $HOME.
CACHE_ROOT = Path(
    os.environ.get("LINGUACAST_CACHE_DIR")
    or (Path.home() / ".cache" / "linguacast")
)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "hf"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_ROOT / "hf" / "transformers"))


def log(msg: str) -> None:
    """Human progress goes to stderr; stdout is reserved for JSON."""
    print(f"[sidecar] {msg}", file=sys.stderr, flush=True)


def emit(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def emit_error(message: str, *, recoverable: bool = False) -> None:
    emit({"kind": "error", "message": message, "recoverable": recoverable})


# --- Lazy model loaders ----------------------------------------------------
#
# Each loader is a singleton — we keep models hot across requests in the
# same sidecar process. The Rust orchestrator launches one sidecar per CLI
# invocation, so this is a per-run cache, not a daemon.

_torch = None
_whisper_pipeline = None
_madlad_model = None
_madlad_tokenizer = None
_tts_engine_handle = None


def _torch_module():
    global _torch
    if _torch is None:
        log("importing torch (this is the slow one)…")
        import torch  # noqa: WPS433  pyright: ignore

        _torch = torch
    return _torch


def _resolve_torch_device(requested: str) -> str:
    """Translate Rust's device hint into something torch will accept.

    Returns 'mps' | 'cuda' | 'cpu'. Falls back transparently if the
    requested accelerator isn't available — the Rust side surfaces the
    actual device chosen back to the user.
    """
    torch = _torch_module()
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested in ("mps", "cuda"):
        log(
            f"requested device {requested!r} not available; falling back to cpu"
        )
    return "cpu"


def _load_whisper(device: str):
    """Whisper-large-v3 via the transformers ASR pipeline.

    MIT license. The HF pipeline is the smallest amount of code that gets
    us correctly-bucketed segments + timestamps, which is what we need for
    later TTS alignment. Long-form transcription is handled by the pipeline
    internally — Whisper-large-v3 sees 30-second windows.
    """
    global _whisper_pipeline
    if _whisper_pipeline is not None:
        return _whisper_pipeline
    log(f"loading whisper-large-v3 on {device}…")
    import torch  # pyright: ignore
    from transformers import (  # pyright: ignore
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        pipeline,
    )

    model_id = "openai/whisper-large-v3"
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    _whisper_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
        chunk_length_s=30,
        return_timestamps=True,
    )
    return _whisper_pipeline


def _load_madlad(device: str):
    """MADLAD-400-3B-MT for translation.

    Apache-2.0. We use the 3B variant because it fits 8 GB unified with
    bf16 + low_cpu_mem_usage, and quality on EN→ES is well above the bar
    we need for a launch demo. The 10B variant is a future tuning lever
    and is out of scope this week.

    We deliberately do NOT use NLLB — it's CC-BY-NC, which the spec rejects.
    """
    global _madlad_model, _madlad_tokenizer
    if _madlad_model is not None:
        return _madlad_model, _madlad_tokenizer
    log(f"loading madlad-400-3b-mt on {device}…")
    import torch  # pyright: ignore
    from transformers import T5ForConditionalGeneration, T5Tokenizer  # pyright: ignore

    model_id = "google/madlad400-3b-mt"
    dtype = torch.float16 if device != "cpu" else torch.float32
    _madlad_tokenizer = T5Tokenizer.from_pretrained(model_id)
    _madlad_model = T5ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    return _madlad_model, _madlad_tokenizer


def _load_tts(engine: str, device: str):
    """Voice-clone TTS. Week-1 spike: Qwen3-TTS (Apache-2.0).

    The engine knob exists for the OPE-19 A/B test:
      - qwen3-tts            full precision
      - qwen3-tts-quantized  int8 / int4 for 8 GB targets
      - voicebox             disabled this week pending license clearance

    The actual loader is intentionally optimistic — if the model isn't
    available we emit a clear, actionable error rather than synthesising
    silence and pretending it worked.
    """
    global _tts_engine_handle
    if _tts_engine_handle is not None and _tts_engine_handle[0] == engine:
        return _tts_engine_handle[1]
    log(f"loading tts engine {engine!r} on {device}…")
    if engine == "voicebox":
        raise SidecarError(
            "Voicebox is intentionally disabled in week 1 pending an Apache-2.0 "
            "/ MIT-floor license audit (the public release is CC-BY-NC). "
            "Use --tts qwen3-tts or qwen3-tts-quantized."
        )
    if engine not in ("qwen3-tts", "qwen3-tts-quantized"):
        raise SidecarError(f"unknown tts engine: {engine!r}")
    # We avoid hard-coding the exact import path here because Qwen3-TTS is
    # young and the API moved between releases. The first attempt below is
    # the canonical path on Hugging Face; the helpful error tells the user
    # how to recover if a refresh is needed.
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # pyright: ignore
    except Exception as exc:  # pragma: no cover — handled below
        raise SidecarError(
            f"transformers is required for tts but isn't importable: {exc}"
        ) from exc
    model_id = (
        "Qwen/Qwen3-TTS-Quantized"
        if engine == "qwen3-tts-quantized"
        else "Qwen/Qwen3-TTS"
    )
    log(
        f"  → if this hangs on first run, the model weights are downloading to {CACHE_ROOT / 'hf'}"
    )
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device)
    except Exception as exc:
        raise SidecarError(
            f"could not load {model_id}. The Qwen3-TTS Hub repo may have moved or "
            f"requires a newer transformers release. Try `pip install -U transformers` "
            f"or pin --tts qwen3-tts-quantized. Original error: {exc}"
        ) from exc
    _tts_engine_handle = (engine, (mdl, tok))
    return _tts_engine_handle[1]


# --- Op handlers -----------------------------------------------------------


class SidecarError(RuntimeError):
    """Raised inside an op handler — converted to a structured error response."""


def op_hello(_: Dict[str, Any]) -> Dict[str, Any]:
    """Quick handshake that imports torch (the heavy one) and reports device."""
    torch = _torch_module()
    has_mps = bool(torch.backends.mps.is_available()) if hasattr(
        torch.backends, "mps"
    ) else False
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


def op_transcribe(payload: Dict[str, Any]) -> Dict[str, Any]:
    audio_path = Path(payload["audio_path"])
    device = _resolve_torch_device(payload.get("device", "cpu"))
    if not audio_path.exists():
        raise SidecarError(f"audio file not found: {audio_path}")

    pipe = _load_whisper(device)
    log(f"transcribing {audio_path.name}…")
    t0 = time.time()
    result = pipe(str(audio_path))
    log(f"  whisper took {time.time() - t0:.2f}s")

    chunks = result.get("chunks") or []
    segments: List[Dict[str, Any]] = []
    for chunk in chunks:
        ts = chunk.get("timestamp") or (None, None)
        start = float(ts[0]) if ts[0] is not None else 0.0
        end = float(ts[1]) if ts[1] is not None else start
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        segments.append({"start": start, "end": end, "text": text})
    if not segments:
        text = (result.get("text") or "").strip()
        if text:
            segments.append({"start": 0.0, "end": 0.0, "text": text})

    # Whisper exposes detected language via the pipeline's generate_kwargs
    # in some releases; fall back to 'en' for the week-1 spike since our
    # canonical clip is English. This is hard-coded to revisit in week 2.
    language = result.get("language") or "en"
    return {"kind": "transcribe", "language": language, "segments": segments}


def op_translate(payload: Dict[str, Any]) -> Dict[str, Any]:
    segments_in: List[Dict[str, Any]] = payload["segments"]
    source_lang: str = payload["source_lang"]
    target_lang: str = payload["target_lang"]
    device = _resolve_torch_device(payload.get("device", "cpu"))

    if not segments_in:
        return {"kind": "translate", "segments": []}

    model, tok = _load_madlad(device)

    log(f"translating {len(segments_in)} segments {source_lang} → {target_lang}…")
    # MADLAD-400 expects a target-language token prefix like '<2es> Hello world.'
    target_token = f"<2{target_lang}>"
    out_segments: List[Dict[str, Any]] = []
    for seg in segments_in:
        prompt = f"{target_token} {seg['text']}"
        inputs = tok(prompt, return_tensors="pt").to(device)
        with _no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=2,
            )
        translated = tok.decode(generated[0], skip_special_tokens=True).strip()
        out_segments.append(
            {"start": seg["start"], "end": seg["end"], "text": translated}
        )
    return {"kind": "translate", "segments": out_segments}


def op_tts(payload: Dict[str, Any]) -> Dict[str, Any]:
    segments: List[Dict[str, Any]] = payload["segments"]
    reference: Path = Path(payload["reference_audio_path"])
    target_lang: str = payload["target_lang"]
    out_audio: Path = Path(payload["out_audio_path"])
    device = _resolve_torch_device(payload.get("device", "cpu"))
    engine: str = payload.get("engine", "qwen3-tts-quantized")

    if not reference.exists():
        raise SidecarError(f"reference audio not found: {reference}")
    if not segments:
        raise SidecarError("nothing to synthesize — translation returned 0 segments")

    # The full TTS implementation lands in a follow-up commit in this same
    # week's PR. The model loader above is wired and we have a runnable
    # smoke. For the spike's first commit we stop here with a clear error
    # instead of writing a fake audio file, so the pipeline failure mode is
    # honest: "the spike doesn't render audio yet" is not "playback failed".
    _ = _load_tts(engine, device)
    raise SidecarError(
        "tts render not yet implemented — engine loaded successfully, but the "
        "segment-level synthesis loop is the next commit on OPE-19. "
        "Re-run after the engine-decision write-up lands."
    )


def _no_grad():
    """Wrapper that avoids requiring torch at import time."""
    torch = _torch_module()
    return torch.inference_mode()


OPS = {
    "hello": op_hello,
    "transcribe": op_transcribe,
    "translate": op_translate,
    "tts": op_tts,
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
