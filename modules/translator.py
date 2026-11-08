"""Lazy-import wrappers for translator to allow lightweight tests."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

_MARIAN_MODELS = None
_loaded_models: Dict[str, Dict[str, Any]] = {}


def _get_models():
    global _MARIAN_MODELS
    if _MARIAN_MODELS is None:
        from config.model_config import MARIAN_MODELS

        _MARIAN_MODELS = MARIAN_MODELS
    return _MARIAN_MODELS


def dependency_error() -> Optional[str]:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import sentencepiece  # noqa: F401
        return None
    except Exception as exc:
        return (
            f"{exc.__class__.__name__}: {exc}. Install PyTorch, Transformers, and SentencePiece in the "
            "same Python environment used by Streamlit (`python -m streamlit run app.py`). "
            "See docs/installation_guide.md."
        )


def _device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_marian(language: str, token: Optional[str] = None) -> Dict[str, Any]:
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    from modules.hf_auth import get_huggingface_token

    models = _get_models()
    if language not in models:
        raise ValueError(f"Unsupported language: {language}")
    if language in _loaded_models:
        return _loaded_models[language]
    model_name = models[language]
    tok = token if token is not None else get_huggingface_token()
    kwargs = {"token": tok} if tok else {}
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name, **kwargs)
        model = MarianMTModel.from_pretrained(model_name, **kwargs)
    except TypeError:
        kwargs = {"use_auth_token": tok} if tok else {}
        tokenizer = MarianTokenizer.from_pretrained(model_name, **kwargs)
        model = MarianMTModel.from_pretrained(model_name, **kwargs)
    dev = _device()
    model.to(dev)
    model.eval()
    _loaded_models[language] = {"tokenizer": tokenizer, "model": model, "model_name": model_name, "device": dev}
    return _loaded_models[language]


def _result(success: bool, **kwargs) -> Dict[str, Any]:
    return {
        "success": success,
        "id": kwargs.get("id"),
        "original": kwargs.get("original"),
        "translated": kwargs.get("translated"),
        "translated_text": kwargs.get("translated_text"),
        "target_language": kwargs.get("target_language"),
        "model": kwargs.get("model"),
        "device": kwargs.get("device"),
        "latency_seconds": kwargs.get("latency_seconds"),
        "load_seconds": kwargs.get("load_seconds"),
        "error": kwargs.get("error"),
    }


def translate_text(text: str, target_language: str = "Hindi", token: Optional[str] = None) -> Dict[str, Any]:
    original = text if isinstance(text, str) else str(text)
    models = _get_models()
    if not isinstance(text, str) or not text.strip():
        return _result(False, original=original, target_language=target_language, latency_seconds=0.0,
                       error="Please enter valid text (non-empty).")
    if target_language not in models:
        return _result(False, original=original, target_language=target_language, latency_seconds=0.0,
                       error=f"Unsupported target language: {target_language}")
    dep = dependency_error()
    if dep:
        return _result(False, original=original, target_language=target_language, latency_seconds=0.0, error=dep)
    try:
        import torch

        load_start = time.perf_counter()
        bundle = load_marian(target_language, token=token)
        load_seconds = round(time.perf_counter() - load_start, 3)
        tokenizer, model, dev = bundle["tokenizer"], bundle["model"], bundle["device"]
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        if dev == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=256, num_beams=4)
        if dev == "cuda":
            torch.cuda.synchronize()
        latency = round(time.perf_counter() - start, 4)
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if not translated:
            return _result(False, original=original, target_language=target_language, model=bundle["model_name"],
                           device=dev, latency_seconds=latency, load_seconds=load_seconds,
                           error="Model returned an empty translation.")
        return _result(True, original=original, translated=translated, translated_text=translated,
                       target_language=target_language, model=bundle["model_name"], device=dev,
                       latency_seconds=latency, load_seconds=load_seconds)
    except Exception as exc:
        # Memory-specific hint
        msg = f"{exc.__class__.__name__}: {exc}"
        if "out of memory" in str(exc).lower() or "cuda" in str(exc).lower() and "memory" in str(exc).lower():
            msg += " Try CPU mode or close other GPU applications (RTX 4050 has 6 GB VRAM)."
        return _result(False, original=original, target_language=target_language, latency_seconds=None, error=msg)


def unload_models() -> None:
    global _loaded_models
    _loaded_models = {}
    from services.model_manager import release_model_memory

    release_model_memory()
