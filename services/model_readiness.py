"""Model-readiness checks for the dashboard.

Reports whether each notebook model can load/run in the current environment
without faking success. Each check returns a dict with:
  status: 'ready' | 'blocked' | 'unknown'
  detail: human-readable reason
  model: HF model id
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Dict


def _check_and_unload(
    check: Callable[[], Dict[str, Any]],
    unload: Callable[[], None],
) -> Dict[str, Any]:
    try:
        return check()
    finally:
        unload()


def _try(mod_name: str) -> bool:
    try:
        __import__(mod_name)
        return True
    except Exception:
        return False


def _gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def check_hf_auth() -> Dict[str, Any]:
    try:
        from modules.hf_auth import get_huggingface_token
        return {
            "status": "ready" if get_huggingface_token() else "not_authenticated",
            "detail": "HF token present in session" if get_huggingface_token() else "No HF token in session (public models may still work).",
        }
    except Exception as exc:
        return {"status": "blocked", "detail": f"hf_auth import failed: {exc}"}


def check_marianmt() -> Dict[str, Any]:
    deps_ok = _try("torch") and _try("transformers") and _try("sentencepiece")
    if not deps_ok:
        return {"status": "blocked", "detail": "PyTorch/Transformers/SentencePiece not installed."}
    try:
        from modules.translator import translate_text
        r = translate_text("Hello.", "Hindi")
        return {
            "status": "ready" if r.get("success") else "blocked",
            "detail": r.get("error") or "Real MarianMT Hindi inference succeeded.",
            "device": r.get("device"),
            "latency_seconds": r.get("latency_seconds"),
        }
    except Exception as exc:
        return {"status": "blocked", "detail": f"MarianMT check raised: {exc}"}


def check_qwen() -> Dict[str, Any]:
    deps_ok = _try("torch") and _try("transformers")
    if not deps_ok:
        return {"status": "blocked", "detail": "PyTorch/Transformers not installed."}
    gpu = _gpu_available()
    try:
        from modules.llm_translator import translate_with_llm
        r = translate_with_llm("Hello.", target_language="Hindi", max_new_tokens=16)
        return {
            "status": "ready" if r.get("success") else "blocked",
            "detail": r.get("error") or "Real Qwen Hindi inference succeeded.",
            "device": r.get("device"),
            "load_mode": r.get("load_mode"),
            "latency_seconds": r.get("latency_seconds"),
            "gpu_available": gpu,
        }
    except Exception as exc:
        return {"status": "blocked", "detail": f"Qwen check raised: {exc}", "gpu_available": gpu}


def check_blip() -> Dict[str, Any]:
    deps_ok = _try("torch") and _try("transformers") and _try("PIL")
    if not deps_ok:
        return {"status": "blocked", "detail": "PyTorch/Transformers/Pillow not installed."}
    try:
        from modules.image_caption import caption_image
        from PIL import Image
        img = Image.new("RGB", (224, 224), color=(120, 160, 90))
        r = caption_image(img)
        return {
            "status": "ready" if r.get("success") else "blocked",
            "detail": r.get("error") or "Real BLIP captioning succeeded.",
            "device": r.get("device"),
            "latency_seconds": r.get("latency_seconds"),
        }
    except Exception as exc:
        return {"status": "blocked", "detail": f"BLIP check raised: {exc}"}


def check_parler_tts() -> Dict[str, Any]:
    deps_ok = _try("torch") and _try("parler_tts") and _try("soundfile")
    if not deps_ok:
        return {"status": "blocked", "detail": "PyTorch/Parler-TTS/soundfile not installed."}
    try:
        from modules.speech import generate_speech
        r = generate_speech("नमस्ते.", language="Hindi", speaker="Divya", style="Normal")
        return {
            "status": "ready" if r.get("success") else "blocked",
            "detail": r.get("error") or "Real Parler-TTS Hindi WAV generation succeeded.",
            "device": r.get("device"),
            "audio_path": r.get("audio_path"),
            "latency_seconds": r.get("latency_seconds"),
        }
    except Exception as exc:
        return {"status": "blocked", "detail": f"Parler-TTS check raised: {exc}"}


def check_all() -> Dict[str, Dict[str, Any]]:
    from modules.image_caption import unload_blip
    from modules.llm_translator import unload_llm
    from modules.speech import unload_tts
    from modules.translator import unload_models

    return {
        "marianmt": _check_and_unload(check_marianmt, unload_models),
        "qwen": _check_and_unload(check_qwen, unload_llm),
        "blip": _check_and_unload(check_blip, unload_blip),
        "parler_tts": _check_and_unload(check_parler_tts, unload_tts),
        "hf_auth": check_hf_auth(),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(check_all(), indent=2, ensure_ascii=False))
